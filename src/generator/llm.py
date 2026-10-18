import logging
import re
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from groq import Groq
from src.config import settings
from src.embeddings.embedder import get_embedder
from src.vector_stores import get_vector_store, SearchResult

logger = logging.getLogger("rag_generator")

FALLBACK_ANSWER = "I do not have sufficient information in the provided context to answer this question."

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
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float          # vector search only (query embedding excluded)
    generation_latency_ms: float
    total_latency_ms: float
    token_usage: Dict[str, int]
    store_type: str
    model_used: str
    # "llm" = grounded LLM answer | "no_context" = refused before calling the LLM |
    # "extractive_fallback" = LLM unavailable, raw top chunk returned (never scored as an LLM answer)
    generation_mode: str = "llm"
    top_score: float = 0.0
    raw_chunks: List[SearchResult]

SYSTEM_PROMPT = f"""You are an assistant that answers questions strictly from the provided context chunks.

RULES:
1. Use ONLY facts stated in the Context Chunks. Do not use outside knowledge, do not extrapolate.
2. After every claim, cite the chunk it came from as [Chunk-N] (N is the chunk number shown).
3. If the chunks do not contain the answer, reply EXACTLY with: "{FALLBACK_ANSWER}" and nothing else.
4. Be concise: answer in 1-3 sentences.
"""

USER_PROMPT_TEMPLATE = """Context Chunks:
{context_block}

User Question: {question}

Grounded Answer (cite chunks like [Chunk-1]):"""

_CITATION_RE = re.compile(r"\[?Chunk-(\d+)\]?", re.IGNORECASE)

def is_fallback_answer(answer: str) -> bool:
    """True only when the answer IS the refusal sentence, not when it merely quotes it
    (e.g. a correct answer to "what phrase is returned when context is insufficient?")."""
    core = re.sub(r"\[?Chunk-\d+\]?", "", answer, flags=re.I).strip().strip("\"'` .").lower()
    return core == FALLBACK_ANSWER.strip(" .").lower()

def extract_cited_indices(answer: str, n_chunks: int) -> List[int]:
    """0-based indices of the chunks the answer cites. Exact match on the number, so 'Chunk-1' never matches 'Chunk-10'."""
    seen: List[int] = []
    for m in _CITATION_RE.finditer(answer):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_chunks and idx not in seen:
            seen.append(idx)
    return seen

class GroundedRAGService:
    def __init__(self, store_type: Optional[str] = None):
        self.store_type = store_type or settings.default_vector_store
        self.embedder = get_embedder()
        self.groq_client = Groq(api_key=settings.groq_api_key, max_retries=2, timeout=30.0) if settings.groq_configured else None

    # Tried in order if the configured model is not available to the API key (Groq retires models).
    FALLBACK_MODELS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile",
                       "llama-3.1-8b-instant")

    def _resolve_model(self) -> str:
        """Configured model if usable, else the first available fallback (looked up once, cached, logged loudly)."""
        if getattr(self, "_model", None):
            return self._model
        self._model = settings.groq_model
        return self._model

    def _available_models(self) -> set:
        if getattr(self, "_avail", None) is None:
            try:
                self._avail = {m.id for m in self.groq_client.models.list().data}
            except Exception as exc:
                logger.error("Cannot list Groq models: %s", exc)
                return set()
        return self._avail

    def _fallback_candidates(self, tried: set) -> List[str]:
        available = self._available_models()
        return [m for m in self.FALLBACK_MODELS if m in available and m not in tried]

    def _chat(self, model: str, user_prompt: str):
        kwargs: Dict[str, Any] = dict(
            model=model, temperature=0.0, max_tokens=2000,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        )
        if model.startswith("qwen/"):
            kwargs["reasoning_format"] = "hidden"   # keep chain-of-thought out of the answer
        elif model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "low"
        try:
            return self.groq_client.chat.completions.create(**kwargs)
        except Exception as exc:
            if type(exc).__name__ == "BadRequestError" and ("reasoning_format" in kwargs or "reasoning_effort" in kwargs):
                kwargs.pop("reasoning_format", None)
                kwargs.pop("reasoning_effort", None)   # model does not accept the reasoning knob: retry plain
                return self.groq_client.chat.completions.create(**kwargs)
            raise

    def _generate(self, user_prompt: str) -> Dict[str, Any]:
        """Calls Groq. If the model is retired (NotFound) or rate-limited, fails over to the next available model.
        A NotFound model becomes the new default for this process; a rate limit only affects this one request."""
        tried: set = set()
        model = self._resolve_model()
        while True:
            tried.add(model)
            try:
                resp = self._chat(model, user_prompt)
                break
            except Exception as exc:
                kind = type(exc).__name__
                if kind not in ("NotFoundError", "RateLimitError"):
                    raise
                candidates = self._fallback_candidates(tried)
                if not candidates:
                    raise
                logger.warning("Model '%s' failed (%s); trying '%s'", model, kind, candidates[0])
                if kind == "NotFoundError":
                    self._model = candidates[0]
                model = candidates[0]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        text = re.sub(r"<think>.*?</think>", "", resp.choices[0].message.content or "", flags=re.S).strip()
        if not text:
            raise RuntimeError(f"Model '{model}' returned an empty answer")
        return {"text": text, "usage": usage, "model": model}

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        store_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> QueryResponse:
        query_text = (query_text or "").strip()
        if not query_text:
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        threshold = settings.similarity_threshold if similarity_threshold is None else similarity_threshold

        total_start = time.perf_counter()
        target_store_type = store_type or self.store_type
        store = get_vector_store(target_store_type)

        # 1. Embed query (timed separately from the vector search)
        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(query_text)
        embed_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Vector search + metadata filter
        t1 = time.perf_counter()
        retrieved = store.search(query_vector=query_vector, top_k=top_k, metadata_filter=metadata_filter)
        search_ms = (time.perf_counter() - t1) * 1000.0
        top_score = retrieved[0].score if retrieved else 0.0

        def build(answer: str, mode: str, citations: List[Citation], chunks: List[SearchResult],
                  usage: Optional[Dict[str, int]] = None, gen_ms: float = 0.0, model: str = "") -> QueryResponse:
            return QueryResponse(
                query=query_text, answer=answer, citations=citations,
                retrieved_chunks_count=len(chunks),
                embedding_latency_ms=round(embed_ms, 2),
                retrieval_latency_ms=round(search_ms, 2),
                generation_latency_ms=round(gen_ms, 2),
                total_latency_ms=round((time.perf_counter() - total_start) * 1000.0, 2),
                token_usage=usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                store_type=target_store_type, model_used=model, generation_mode=mode,
                top_score=round(top_score, 4), raw_chunks=chunks,
            )

        # 3. "No relevant context" gate: refuse BEFORE calling the LLM when the best chunk is too dissimilar.
        #    The gate keys on the top score so one weak tail chunk does not trigger a refusal,
        #    and chunks below the threshold are not passed to the LLM.
        usable = [c for c in retrieved if c.score >= threshold]
        if not usable:
            return build(FALLBACK_ANSWER, "no_context", [], [])

        # 4. Grounded prompt
        context_block = "\n\n".join(
            f"[Chunk-{i + 1}] (File: {c.filename}, Section: {c.section}):\n{c.text.strip()}"
            for i, c in enumerate(usable)
        )
        user_prompt = USER_PROMPT_TEMPLATE.format(context_block=context_block, question=query_text)

        # 5. Generate
        if self.groq_client is None:
            return self._extractive(build, usable, "GROQ_API_KEY is not configured")

        g0 = time.perf_counter()
        try:
            out = self._generate(user_prompt)
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            return self._extractive(build, usable, f"LLM call failed ({type(exc).__name__})")
        gen_ms = (time.perf_counter() - g0) * 1000.0
        answer = out["text"]

        # 6. Citations = chunks the model actually cited (none if it refused)
        citations: List[Citation] = []
        if not is_fallback_answer(answer):
            idxs = extract_cited_indices(answer, len(usable))
            citations = [
                Citation(chunk_id=usable[i].chunk_id, filename=usable[i].filename,
                         section=usable[i].section, score=usable[i].score)
                for i in idxs
            ]
        return build(answer, "llm", citations, usable, out["usage"], gen_ms, out.get("model", settings.groq_model))

    @staticmethod
    def _extractive(build, usable: List[SearchResult], reason: str) -> QueryResponse:
        """Degraded mode when the LLM is unavailable: return the best chunk verbatim, clearly labelled."""
        top = usable[0]
        answer = (f"[LLM unavailable: {reason}. Showing the most relevant passage verbatim.]\n\n"
                  f"{top.text.strip()}")
        return build(answer, "extractive_fallback",
                     [Citation(chunk_id=top.chunk_id, filename=top.filename, section=top.section, score=top.score)],
                     usable)
