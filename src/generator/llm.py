import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from groq import Groq
from src.config import settings
from src.embeddings.embedder import get_embedder
from src.vector_stores import get_vector_store, SearchResult

class Citation(BaseModel):
    chunk_id: str
    filename: str
    section: str
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieved_chunks_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    token_usage: Dict[str, int]
    store_type: str
    model_used: str
    raw_chunks: List[SearchResult]

SYSTEM_PROMPT = """You are an expert AI engineering assistant specializing in Vector Databases, RAG architectures, and Cloud Retrieval Systems.
Your task is to answer user questions with strict grounding in the provided context chunks.

RULES:
1. Grounding & Faithfulness: Rely ONLY on the clear facts directly mentioned in the Context Chunks below. Do NOT assume, extrapolate, or use outside knowledge.
2. Citations: Whenever you state a claim or fact, cite the chunk it came from using the format [Chunk-X] or the chunk ID (e.g. [Chunk-1]).
3. Fallback on Insufficient Context: If the provided chunks do not contain the answer, or if the question cannot be answered purely from the text, reply EXACTLY with:
"I do not have sufficient information in the provided context to answer this question."
Do NOT hallucinate, guess, or attempt to be helpful with ungrounded answers.
"""

USER_PROMPT_TEMPLATE = """Context Chunks:
{context_block}

User Question: {question}

Grounded Answer (remember to cite chunks like [Chunk-1]):"""

class GroundedRAGService:
    def __init__(self, store_type: Optional[str] = None):
        self.store_type = store_type or settings.default_vector_store
        self.embedder = get_embedder()
        self.groq_client = None
        if settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here":
            self.groq_client = Groq(api_key=settings.groq_api_key)

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        store_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.25
    ) -> QueryResponse:
        total_start = time.perf_counter()
        target_store_type = store_type or self.store_type
        store = get_vector_store(target_store_type)

        # 1. Embed query
        retrieval_start = time.perf_counter()
        query_vector = self.embedder.embed_query(query_text)

        # 2. Vector search with metadata filter
        retrieved_chunks = store.search(
            query_vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Filter by minimum relevance threshold
        filtered_chunks = [c for c in retrieved_chunks if c.score >= similarity_threshold]

        # 3. Handle 'no relevant context' before LLM call if nothing meets threshold
        if not filtered_chunks:
            total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            return QueryResponse(
                query=query_text,
                answer="I do not have sufficient information in the provided context to answer this question.",
                citations=[],
                retrieved_chunks_count=0,
                retrieval_latency_ms=round(retrieval_elapsed_ms, 2),
                generation_latency_ms=0.0,
                total_latency_ms=round(total_elapsed_ms, 2),
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                store_type=target_store_type,
                model_used=settings.groq_model,
                raw_chunks=[]
            )

        # 4. Construct grounded prompt with citations
        context_parts = []
        for idx, chunk in enumerate(filtered_chunks):
            context_parts.append(
                f"[Chunk-{idx + 1}] (ID: {chunk.chunk_id}, File: {chunk.filename}, Section: {chunk.section}):\n{chunk.text.strip()}"
            )
        context_block = "\n\n".join(context_parts)
        user_prompt = USER_PROMPT_TEMPLATE.format(context_block=context_block, question=query_text)

        # 5. Generate LLM Answer
        gen_start = time.perf_counter()
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answer_text = ""
        actual_model_used = settings.groq_model

        if self.groq_client:
            accessible_models = []
            model_discovery_err = None
            try:
                model_objs = self.groq_client.models.list()
                accessible_models = [m.id for m in model_objs.data]
            except Exception as e:
                model_discovery_err = str(e)

            # Determine models to try based on account permissions
            candidate_preferences = [
                settings.groq_model,
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-3.2-3b-preview",
                "llama-3.2-1b-preview",
                "llama-3.2-11b-vision-preview",
                "llama-3.2-90b-vision-preview",
                "gemma2-9b-it"
            ]

            if accessible_models:
                models_to_try = [m for m in candidate_preferences if m in accessible_models]
                if not models_to_try:
                    # Pick any accessible chat/text model
                    models_to_try = [m for m in accessible_models if not any(x in m.lower() for x in ["whisper", "guard", "embed"])]
            else:
                models_to_try = candidate_preferences[:4]

            errors = []
            if model_discovery_err:
                errors.append(f"Model discovery failed: {model_discovery_err}")

            for model_name in models_to_try:
                try:
                    response = self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=800
                    )
                    answer_text = response.choices[0].message.content or ""
                    if response.usage:
                        token_usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    actual_model_used = model_name
                    errors = []
                    break
                except Exception as e:
                    try:
                        response = self.groq_client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}
                            ],
                            max_tokens=800
                        )
                        answer_text = response.choices[0].message.content or ""
                        if response.usage:
                            token_usage = {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            }
                        actual_model_used = model_name
                        errors = []
                        break
                    except Exception as e2:
                        errors.append(f"{model_name}: {str(e)}")
                        continue

            if errors and not answer_text:
                top_c = filtered_chunks[0]
                available_str = ", ".join(accessible_models) if accessible_models else "None accessible"
                answer_text = (
                    f"Based on [{top_c.filename} - {top_c.section}]:\n\n"
                    f"{top_c.text.strip()}\n\n"
                    f"[Chunk-1]\n\n"
                    f"⚠️ Note: Groq error ({errors[0].splitlines()[0]}). Accessible models: [{available_str}]. "
                    f"Please verify your GROQ_API_KEY in Render."
                )
        else:
            top_c = filtered_chunks[0]
            answer_text = (
                f"Based on [{top_c.filename} - {top_c.section}]:\n\n"
                f"{top_c.text.strip()}\n\n"
                f"[Chunk-1]\n\n"
                f"⚠️ Note: GROQ_API_KEY is not configured. Set your free key from https://console.groq.com/keys."
            )

        gen_elapsed_ms = (time.perf_counter() - gen_start) * 1000.0
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        # 6. Extract citations
        citations = []
        for idx, chunk in enumerate(filtered_chunks):
            chunk_tag = f"Chunk-{idx + 1}"
            if chunk_tag in answer_text or chunk.chunk_id in answer_text or len(filtered_chunks) == 1:
                citations.append(Citation(
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    section=chunk.section,
                    score=chunk.score
                ))

        # If model answered from chunks but omitted explicit tags, cite the top relevant chunks used
        if not citations and "insufficient information" not in answer_text.lower():
            citations = [
                Citation(
                    chunk_id=c.chunk_id,
                    filename=c.filename,
                    section=c.section,
                    score=c.score
                )
                for c in filtered_chunks[:2]
            ]

        return QueryResponse(
            query=query_text,
            answer=answer_text.strip(),
            citations=citations,
            retrieved_chunks_count=len(filtered_chunks),
            retrieval_latency_ms=round(retrieval_elapsed_ms, 2),
            generation_latency_ms=round(gen_elapsed_ms, 2),
            total_latency_ms=round(total_elapsed_ms, 2),
            token_usage=token_usage,
            store_type=target_store_type,
            model_used=actual_model_used,
            raw_chunks=filtered_chunks
        )
