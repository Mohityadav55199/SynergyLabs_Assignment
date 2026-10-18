"""LLM-as-judge for answer quality (faithfulness + answer relevance + correctness vs gold).

* Judge model = JUDGE_MODEL (default: a different family from the Llama generator, to limit self-enhancement bias).
* Faithfulness is claim-level: the judge lists every factual claim in the answer and marks each SUPPORTED /
  UNSUPPORTED by the retrieved context only -> score = supported / total claims. This is more auditable than a
  single holistic number.
* Every prompt and raw response is appended to evaluation/judge_log.jsonl (replayable).
"""
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq

from src.config import settings

LOG_PATH = Path(__file__).resolve().parent / "judge_log.jsonl"

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of retrieval-augmented answers. "
    "Judge ONLY from the material given. Respond with a single JSON object and nothing else."
)

JUDGE_TEMPLATE = """QUESTION:
{question}

RETRIEVED CONTEXT (the only evidence the answerer was allowed to use):
{context}

GOLD ANSWER (reference written by a human):
{gold}

ANSWER TO EVALUATE:
{answer}

Tasks:
1. "claims": split the ANSWER into atomic factual claims (ignore citation tags like [Chunk-1]). For each claim give
   {{"claim": str, "supported": true|false}}; supported=true ONLY if the RETRIEVED CONTEXT states or directly entails it
   (a claim true in the real world but absent from the context is UNSUPPORTED).
2. "relevance": integer 1-5 - does the ANSWER address what the QUESTION asks? 1 = off-topic, 3 = partly, 5 = fully and directly.
3. "correctness": integer 1-5 compared with the GOLD ANSWER. 1 = contradicts/misses it, 3 = partially matches, 5 = same facts.
4. "rationale": one or two sentences.

Return JSON: {{"claims": [...], "relevance": int, "correctness": int, "rationale": str}}"""


def parse_judge_json(raw: str) -> Optional[Dict[str, Any]]:
    """Tolerant parser: strips code fences / reasoning text, takes the outermost JSON object."""
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def score_verdict(v: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Validates a parsed verdict and converts it to 0-1 scores. None if structurally invalid."""
    try:
        claims = v["claims"]
        rel, cor = int(v["relevance"]), int(v["correctness"])
        if not (1 <= rel <= 5 and 1 <= cor <= 5) or not isinstance(claims, list):
            return None
        supported = sum(1 for c in claims if c.get("supported") is True)
        faith = (supported / len(claims)) if claims else None  # no claims (e.g. refusal) -> not applicable
        return {"faithfulness": faith, "relevance": (rel - 1) / 4.0, "correctness": (cor - 1) / 4.0,
                "n_claims": len(claims), "n_supported": supported}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


class AnswerJudge:
    def __init__(self, model: Optional[str] = None):
        if not settings.groq_configured:
            raise RuntimeError("GROQ_API_KEY is not configured; cannot run the LLM judge")
        self.model = model or settings.judge_model
        self.client = Groq(api_key=settings.groq_api_key, max_retries=3, timeout=60.0)
        self.calls = 0
        self.tokens = 0

    def judge(self, case_id: str, question: str, context_chunks: List[str], gold: str, answer: str) -> Dict[str, Any]:
        context = "\n\n".join(f"[Chunk-{i + 1}] {c.strip()}" for i, c in enumerate(context_chunks)) or "(no context retrieved)"
        prompt = JUDGE_TEMPLATE.format(question=question, context=context, gold=gold, answer=answer)
        record: Dict[str, Any] = {"case_id": case_id, "judge_model": self.model, "prompt": prompt, "attempts": []}
        scores = None
        for attempt in range(2):  # one retry on malformed output
            raw, usage = "", 0
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=0.0, max_tokens=1500,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content or ""
                usage = resp.usage.total_tokens if resp.usage else 0
            except Exception as exc:
                raw = f"__ERROR__ {type(exc).__name__}: {exc}"
            self.calls += 1
            self.tokens += usage
            record["attempts"].append({"raw_response": raw, "tokens": usage})
            parsed = parse_judge_json(raw)
            scores = score_verdict(parsed) if parsed else None
            if scores:
                record["rationale"] = parsed.get("rationale")
                break
            time.sleep(1)
        record["scores"] = scores
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"scores": scores, "rationale": record.get("rationale")}
