# RAG Eval — LLM Evals Course TA

A retrieval-augmented generation (RAG) chatbot that answers questions **only** from a course's video transcripts, paired with a full, production-style **evaluation suite** built on [DeepEval](https://github.com/confident-ai/deepeval). The project is really two things in one repo: a small but complete RAG app, and a regression-testing harness that decides — with a machine-checkable verdict — whether a change to that app is safe to ship.

> Ask it something covered in the transcripts and it answers, grounded in the retrieved chunks. Ask it something outside that scope and it says so, instead of making something up.

## Why this exists

Most RAG demos stop at "it works on a few examples." This project treats the RAG pipeline as a system with components that can each fail in their own way — the retriever can miss the right chunk, the generator can hallucinate even with the right chunk, the whole thing can drift out of scope or leak something it shouldn't — and builds a separate, targeted eval for each failure mode. Those evals are then wired into a `baseline` vs `candidate` comparison with **gates** (hard blockers) and **guardrails** (soft warnings), so a change to a prompt, model, or retrieval parameter gets a PASS / REVIEW / FAIL verdict instead of a vibe check.

## Architecture

```
data/*.vtt  ──►  chunk + embed  ──►  Chroma vector store
                                          │
                                          ▼
                          ┌── RETRIEVER (bi-encoder, over-fetch) ──┐
                          │                                        │
                          ▼                                        │
                    RERANKER (cross-encoder, keeps top_k)          │
                          │                                        │
                          ▼                                        │
                    GENERATOR (gpt-4o-mini, faithfulness-first     │
                    prompt — answers only from context, abstains   │
                    otherwise)                                     │
                          │                                        │
                          ▼                                        │
                       answer  ◄──────────── used by ───────────────
```

- **Retriever** (`src/retriever.py`) — loads course `.vtt` transcripts, strips timestamps, chunks them (1000 chars, 150 overlap), embeds with `text-embedding-3-large`, and persists to a local Chroma store.
- **Reranker** (`src/reranker.py`) — over-fetches `fetch_k` candidates from the vector store, then re-scores each `(query, chunk)` pair with a `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder and keeps the top `top_k`.
- **Generator** (`src/generator.py`) — `gpt-4o-mini` behind a long, deliberately hardened prompt: answer only from the provided context, abstain when it can't, stay in the teaching-assistant role, resist prompt-leakage and jailbreak attempts, and never reproduce PII or protected course content verbatim. Ships both a blocking `generate()` and a streaming `generate_stream()` (for time-to-first-token measurement).
- **Pipeline** (`src/rag_pipeline.py`) — wires retriever → reranker → generator into one `RagPipeline.invoke(query)` call that returns `{query, context, answer}`, the shared shape every eval consumes.
- **UI** (`src/app.py`) — a Streamlit chat app with sliders for `fetch_k`/`top_k`, an expandable "retrieved chunks" panel, and abstention messaging.

## The eval suite

Every eval takes the *same* `RagPipeline` (or a component of it) and produces DeepEval `LLMTestCase`s scored by an LLM judge (`gpt-4o-mini`) or by direct measurement. They're grouped into three tiers:

| Tier | Evals | What it checks |
|---|---|---|
| **Component** | `eval_retriever.py`, `eval_generator.py` | Retriever in isolation (contextual recall/precision against a golden set); generator in isolation, fed known-good context (faithfulness, answer relevancy) — so a bad score is unambiguously that component's fault. |
| **Application** | `eval_rag_pipeline.py`, `eval_application.py` | The RAG triad (faithfulness, answer relevancy, contextual relevancy) end-to-end; plus reference-based correctness and completeness via `GEval` rubrics. |
| **Safety** | `eval_scope_safety.py`, `eval_leakage.py`, `eval_toxicity.py`, merged as `eval_safety.py` | Stays in its teaching-assistant scope under jailbreak/roleplay pressure; doesn't leak the system prompt, protected course content, or PII; doesn't produce toxic output. |
| **Operational** | `eval_latency.py`, `eval_cost.py`, `eval_reliability.py`, merged as `eval_ops.py` | End-to-end and time-to-first-token latency (percentiles vs an SLO); derived per-query cost (tokens × price, with cache-aware pricing); success/error/retry rates under a backoff wrapper. These need no golden set or judge — they're direct measurements. |

Shared helpers live in `evals/harness.py` (golden-file loading, per-metric pass-rate/score summaries that stay correct across DeepEval versions).

### Regression testing: baseline vs candidate

This is the part that turns the evals into a CI gate, not just a report.

1. **`evals/run_suite.py`** builds one `RagPipeline`, runs *every* eval tier against it, flattens all results into `namespace.metric.stat` keys, and writes a timestamped, git-SHA-stamped JSON **snapshot**.
   ```bash
   python -m evals.run_suite --baseline --label "k=10 + reranker"   # -> baselines/baseline.json
   python -m evals.run_suite --label "k=15, no reranker"            # -> baselines/candidate.json
   ```
2. **`evals/metric_registry.py`** is the policy file: for every metric it declares whether higher-or-lower is better, whether it's a **gate** (safety — any regression blocks), a **guardrail** (quality/ops — a regression beyond a noise-calibrated tolerance flags for human review), or **info** (tracked, never blocking).
3. **`evals/compare.py`** diffs the two snapshots against those rules and prints a table plus one verdict:
   ```bash
   python -m evals.compare --baseline baselines/baseline.json --candidate baselines/candidate.json
   ```
   - **PASS** — safe to promote (exit code 0)
   - **REVIEW** — a guardrail regressed beyond tolerance; a human decides (exit code 2)
   - **FAIL** — a gate regressed; blocked, no discussion (exit code 1)

   The non-zero exit codes are intentional — this is designed to sit in a CI pipeline as an actual merge gate.

## Project layout

```
src/
  retriever.py     # load transcripts, chunk, embed, persist Chroma store
  reranker.py       # cross-encoder reranking retriever
  generator.py      # faithfulness-first prompt + LLM chain (+ streaming)
  rag_pipeline.py   # retriever -> reranker -> generator
  app.py            # Streamlit chat UI
evals/
  eval_retriever.py, eval_generator.py         # component-level
  eval_rag_pipeline.py, eval_application.py    # application-level
  eval_scope_safety.py, eval_leakage.py,
  eval_toxicity.py, eval_safety.py             # safety (merged runner)
  eval_latency.py, eval_cost.py,
  eval_reliability.py, eval_ops.py             # operational (merged runner)
  metric_registry.py   # gate/guardrail/info rules per metric
  run_suite.py          # orchestrator -> snapshot JSON
  compare.py            # snapshot diff -> PASS/REVIEW/FAIL verdict
  harness.py            # shared golden-loading + summarization helpers
goldens/            # hand-authored + synthesized eval datasets (JSON)
data/               # course .vtt transcripts (source knowledge base)
resources/           # small standalone DeepEval usage examples
export_chroma_chunks.py  # dump the Chroma store to JSON for inspection
```

## Setup

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <this-repo>
cd rag-eval
uv sync
```

Create a `.env` in the project root:

```
OPENAI_API_KEY=sk-...
```

Drop your course transcripts (`.vtt` files) into `data/`. The vector store is built automatically on first run and persisted to `chroma_store/` (gitignored — rebuild anytime by deleting the folder and re-running).

## Running it

**Chat UI:**
```bash
uv run streamlit run src/app.py
```

**Smoke-test the pipeline directly:**
```bash
uv run python -m src.rag_pipeline
```

**Run one eval:**
```bash
uv run python -m evals.eval_rag_pipeline
```

**Run the full suite and get a snapshot:**
```bash
uv run python -m evals.run_suite --baseline --label "initial baseline"
```

**Make a change (prompt, `top_k`, model, retriever, whatever), then check whether it's safe to ship:**
```bash
uv run python -m evals.run_suite --label "describe the change"
uv run python -m evals.compare
```

## Notable design choices

- **Component isolation** — the generator eval feeds it *golden* context, not the retriever's live output, so a low faithfulness score can't be blamed on a bad retrieval.
- **Direction- and kind-aware regression rules** — metrics don't all point the same way (latency/cost/toxicity are lower-is-better; correctness/relevancy are higher-is-better), and don't all matter the same way (a safety regression blocks; a quality wobble gets flagged). The registry encodes both explicitly instead of a single blanket threshold.
- **Tolerances calibrated from measured noise**, not guessed — two runs of the *identical* pipeline were used to measure how much judge scores and latency naturally wobble, and guardrail tolerances are set above that noise floor.
- **Judge metrics gated on average score, not pass rate** — pass rate is threshold-brittle (one borderline case flips it); the mean is the steadier signal for detecting real regressions.
- **Cost modeled honestly** — token counts are pulled straight off the LLM response's usage metadata (by composing `prompt | llm` and stopping short of the string-only output parser), with cache-aware pricing since the large fixed system prompt is a repeated prefix.
- **A hardened generator prompt** — explicit rules against role-breaking, system-prompt leakage, verbatim reproduction of course material, and PII exposure, tested directly by the safety eval tier rather than assumed.
