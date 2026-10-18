# Handover & Runbook

Audience: whoever operates this service after the author. Read this before touching production.

## 1. What runs where

| Piece | What it is | Where | Owner / login | Cost |
|---|---|---|---|---|
| Source code | This repo | GitHub `Mohityadav55199/SynergyLabs_Assignment` | Mohit Yadav (GitHub: Mohityadav55199) | free |
| Backend API | FastAPI + embedded LanceDB/Qdrant + local embedding model | Render web service `srv-db9jnntg1s2s738vlhbg` (`cost-efficient-rag-service`, free plan, Oregon) | **[CONFIRM: Render account email]** | $0 (free tier) |
| Frontend | Static chat UI in `public/` | Vercel project, https://synergylabs-lilac.vercel.app | **[CONFIRM: Vercel account]** | $0 |
| LLM (generation) | Groq-hosted `qwen/qwen3.8-27b` (the service falls back to another available model if this one is retired; see `/llm-status`) | console.groq.com, API key stored ONLY as Render env var `GROQ_API_KEY` | **[CONFIRM: Groq account email]** | free tier / pay-as-you-go |
| LLM (eval judge) | Groq-hosted `openai/gpt-oss-120b`, used only when running the evaluation harness | same Groq key | same | pennies per run |
| Keep-warm pinger | GitHub Actions cron hitting `/health` every 10 min | `.github/workflows/keep-warm.yml` | repo owner | free |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed, runs inside the API process | n/a | n/a | $0 |

There is **no database server**. The vector indexes are files (`data/lancedb`, `data/qdrant`) built from `data/corpus/` during the Render build, so they live and die with the deployed container.

## 2. Environment variables (names only; values never in git)

See `.env.example` for the full annotated list. Production-relevant: `GROQ_API_KEY` (secret), `GROQ_MODEL`, `DEFAULT_VECTOR_STORE`, `SIMILARITY_THRESHOLD`, `CORS_ORIGINS` (set to the Vercel URL), `ADMIN_TOKEN` (secret; protects `/ingest*` and `/llm-status` when set), `MAX_UPLOAD_MB`. Render: Dashboard -> service -> Environment. `render.yaml` marks secrets `sync: false`, so they are never read from git.

## 3. Deploy / redeploy

* **Backend**: push to `main` -> Render auto-builds (`render.yaml`: `pip install`, then `python -m src.cli ingest` for both stores) and starts `uvicorn`. Health check: `GET /health` (`vectors_indexed` should be 100 for the shipped corpus).
* **Frontend**: push to `main` -> Vercel redeploys `public/`. The backend URL is hard-coded in `public/index.html` (`endpoint`); change it there if the Render service is recreated.
* **Local**: `pip install -r requirements-dev.txt`, copy `.env.example` to `.env`, `python -m src.cli ingest`, `python -m src.cli serve`. Tests: `python -m pytest tests`.
* **Docker**: `docker build -t rag . && docker run -p 8000:8000 -e GROQ_API_KEY=... rag`.

## 4. Access and secrets

* Anyone with the Render dashboard login can read/rotate env vars. Rotate `GROQ_API_KEY` at console.groq.com -> API Keys, paste the new value in Render -> Environment -> Save (triggers a restart). Check with `GET /llm-status` (send `X-Admin-Token` if set); it never returns key material.
* **The public API is unauthenticated for `/query`** (by design: demo). `/ingest*` is open unless `ADMIN_TOKEN` is set; **set it before sharing the URL widely** (the demo UI's upload button then needs the header, so upload will stop working from the UI; it is a demo feature).
* The `.env` file is git-ignored. If a key was ever committed or pasted in a chat, treat it as leaked and rotate it.

## 5. What to do when something stops working

| Symptom | Likely cause | Fix |
|---|---|---|
| UI shows the "waking up" banner for 30-60 s, first request slow | Render free tier sleeps after 15 min idle | Expected. The GitHub Actions pinger reduces it; paid plan removes it. |
| Answers start with `[LLM unavailable: ...]`, `generation_mode: extractive_fallback` | Groq key missing/revoked, model retired, rate limit, Groq outage | `GET /llm-status`. Fix key, or set `GROQ_MODEL` to a currently available model from its `available_models` list. The service keeps serving retrieval meanwhile. |
| Every question answered "I do not have sufficient information..." | `SIMILARITY_THRESHOLD` too high for the corpus/embedding model, or index empty | `GET /health` -> `vectors_indexed`. Re-ingest. Lower the threshold (see `similarity_gate` in `evaluation/results_summary.md`). |
| `/health` shows `vectors_indexed: 0` or 5xx on `/query` | Build step failed or container restarted and lost uploaded data | Check Render build logs; Manual Deploy -> "Clear build cache & deploy". |
| Uploaded documents vanished | Render free disk is ephemeral (wiped on every deploy/restart) | Known limitation. Put new documents in `data/corpus/` and commit, or attach a Render persistent disk and point `LANCEDB_DIR`/`QDRANT_DIR`/`CORPUS_DIR` at it. |
| `database is locked` / Qdrant "already accessed by another instance" | Two processes opened the same embedded store (e.g. `--workers 2`, or a CLI run next to the server) | Run exactly one uvicorn worker; stop the other process. |
| Ingest returns 422 for a file | Empty/scanned (image-only) PDF, corrupt file, unsupported type | The `detail` field says which; OCR scanned PDFs before ingesting. |
| Eval harness exits "holds N vectors but corpus yields M" | Index built with different chunk settings than the harness | `python -m src.cli ingest` after clearing `data/lancedb` and `data/qdrant`. |
| Build fails downloading the embedding model | Network/Hugging Face outage | Retry the deploy; the model is fetched once at build time. |

Logs: Render -> Logs. Each query logs one line: `query store=... mode=... top_score=... chunks=N latency_ms(total, embed, search, gen) tokens(prompt, completion, total) citations=N`. Search for `ERROR` / `Groq generation failed`.

## 6. Re-indexing / changing chunking

`python -m src.cli ingest --chunk-size 400 --overlap 40` (or `POST /ingest`). Ingestion is idempotent (SHA-256 of doc id + chunk index + text). If you **change chunk size/overlap on an already-indexed corpus**, old-size chunks have different hashes and will sit beside the new ones: delete `data/lancedb` / `data/qdrant` first. Editing a document replaces its old vectors automatically (same filename, new content hash). After changing chunking, re-run the evaluation and re-calibrate `SIMILARITY_THRESHOLD`.

## 7. Known limitations (be honest with the client)

1. Corpus is small (100 chunks, 6 documents); the evaluation set is 30 questions. Metrics have wide confidence intervals; one question moves hit rate by ~4 points. Relevance labels are derived from a hand-picked evidence string per question, not from human annotation of every chunk.
2. Three of the six documents were written by the author about this very project, which makes questions about them easier than a real corpus would be. The three Wikipedia pages (CC BY-SA 4.0) are a partial counterweight.
3. The similarity threshold (0.68) was tuned on the same 30 questions; the best unanswerable score (0.71) overlaps the weakest answerable score (0.73), so near-domain unanswerable questions rely on the LLM refusing, not on the gate.
4. Latency at 100K / 1M vectors is measured on synthetic vectors on one laptop (`results_scale.json`), not on production hardware or real text.
5. Single node, no replication/HA; one writer at a time; Render free tier sleeps and has ephemeral disk.
6. Local-mode Qdrant is brute-force, not HNSW. It is benchmarked only as a secondary store.
7. PDF section metadata is approximate (page text is re-flowed into paragraphs). Scanned PDFs are not supported (no OCR).
8. Prices in the cost model are a dated snapshot (2026-10-18); two of them (Qdrant Cloud RAM-hour, EC2) come from third-party pages. Re-run `python -m evaluation.cost_model` after updating the constants.
9. The Groq model names can be retired by Groq at any time; there is no automatic model fallback (it fails loudly into extractive mode instead).
