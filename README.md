# RAG Evaluation & Regression Testing with DeepEval

A production-oriented **Retrieval-Augmented Generation (RAG) evaluation framework** that evaluates a RAG application at multiple levels — **retriever, generator, end-to-end pipeline, application quality, safety, and operations** — and then uses **baseline-vs-candidate regression testing** to decide whether a pipeline change is safe to promote.

The project is built around the idea that a RAG system should not be judged only by whether its final answer "looks good". A change to chunking, retrieval, reranking, prompting, or generation can improve one dimension while silently damaging another. This project turns those trade-offs into measurable regression signals.

> **Core idea:** build the RAG pipeline once, evaluate its individual components and the complete application, persist the results as a snapshot, and compare a candidate snapshot against a known-good baseline.

---

## 📸 Evaluation Regression Report

The following is a real evaluation comparison produced by this project:

![RAG evaluation regression comparison](docs/evaluation.png)

### What this run shows

The captured comparison contains:

- **2 regressions**
- **6 improvements**
- **12 flat metrics**
- **1 informational metric**
- Overall result: **REGRESSED**

The most important regressions shown are:

| Metric | Baseline | Candidate | Delta | Interpretation |
|---|---:|---:|---:|---|
| `pipeline.contextual_relevancy.avg_score` | 0.4549 | 0.3863 | -0.06852 | Retrieved context became less relevant |
| `safety.leakage.pii_avg_score` | 0.8 | 0.6 | -0.2 | PII protection became worse |

At the same time, several dimensions improved:

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| `generator.answer_relevancy.avg_score` | 0.9749 | 0.9802 | +0.005243 |
| `generator.faithfulness.avg_score` | 0.9475 | 0.9561 | +0.008675 |
| `pipeline.faithfulness.avg_score` | 0.9459 | 0.9598 | +0.01393 |
| `retriever.contextual_precision.avg_score` | 0.8637 | 0.9011 | +0.03741 |
| `retriever.contextual_recall.avg_score` | 0.8850 | 0.9622 | +0.07722 |
| `ops.cost.cost_per_query_usd` | 0.000273 | 0.0002185 | -0.0000545 |

This is exactly the type of trade-off a regression-testing system should expose: **retrieval and generation can improve while a safety dimension gets worse**.

> Note: the screenshot represents a captured run of the project. The current `evals/compare.py` in this repository uses the verdict names `PASS`, `REVIEW`, and `FAIL`, where a guardrail regression produces `REVIEW` and a gate regression produces `FAIL`.

---

# 1. Project Overview

This project implements an evaluation harness around a course-material RAG assistant.

The RAG application:

1. Loads course transcripts from `.vtt` files.
2. Removes VTT timestamps.
3. Splits transcripts into chunks.
4. Embeds chunks using OpenAI embeddings.
5. Stores vectors in Chroma.
6. Retrieves candidate chunks.
7. Reranks candidates with a cross-encoder.
8. Sends the selected context to an OpenAI chat model.
9. Generates an answer using a faithfulness-first prompt.
10. Evaluates the resulting system using DeepEval and custom operational/safety evaluators.
11. Stores evaluation results in a JSON snapshot.
12. Compares candidate results against a baseline.
13. Classifies changes as improved, flat, regressed, blocked, new, dropped, or informational.

---

# 2. Why This Project?

A normal RAG demo usually asks:

> "Does the chatbot answer the question?"

That is not enough for a production RAG system.

A change such as:

```text
chunk_size: 1000 -> 500
```

can cause:

```text
retrieval recall       ↑
retrieval precision    ↑
answer faithfulness    ↑
answer relevance       ↑
PII protection         ↓
contextual relevancy   ↓
latency                ↑
cost                   ↑
```

If you only inspect the final answer manually, you can easily miss these regressions.

This project therefore evaluates the system across multiple dimensions and treats evaluation as a **regression-testing problem**, similar to software engineering tests.

---

# 3. High-Level Architecture

```text
                         ┌───────────────────────┐
                         │     User Question     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   RAG Application     │
                         │    RagPipeline        │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌──────────────────┐              ┌──────────────────┐
          │    Retriever     │              │    Generator     │
          │                  │              │                  │
          │ OpenAI Embedding │              │ GPT-4o-mini      │
          │ + Chroma         │              │ Grounded Prompt  │
          └────────┬─────────┘              └────────┬─────────┘
                   │                                 │
                   ▼                                 │
          ┌──────────────────┐                       │
          │    Reranker      │                       │
          │                  │                       │
          │ MS-MARCO         │                       │
          │ Cross Encoder    │                       │
          └────────┬─────────┘                       │
                   │                                 │
                   └──────────────┬──────────────────┘
                                  ▼
                           ┌─────────────┐
                           │   Answer    │
                           └──────┬──────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │        Evaluation Suite         │
                  ├─────────────────────────────────┤
                  │ 1. Retriever Evaluation         │
                  │ 2. Generator Evaluation        │
                  │ 3. Pipeline Evaluation         │
                  │ 4. Application Evaluation       │
                  │ 5. Safety Evaluation            │
                  │ 6. Operational Evaluation       │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Evaluation Snapshot │
                         │      JSON           │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
              baseline.json                 candidate.json
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Regression Engine   │
                         │   evals/compare.py  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                       PASS       REVIEW      FAIL
```

---

# 4. Evaluation Strategy

The project deliberately separates evaluation into multiple layers.

## Layer 1 — Component-Level Evaluation

Component-level evaluation isolates individual RAG components so a failure can be attributed to the retriever or generator rather than the whole application.

DeepEval describes RAG evaluation around retriever metrics such as contextual precision, contextual recall and contextual relevancy, and generator metrics such as answer relevancy and faithfulness.

### Retriever

Implemented in:

```text
evals/eval_retriever.py
```

Metrics:

- Contextual Precision
- Contextual Recall

The retriever is evaluated using golden queries and expected answers while the actual retrieval context comes from the live retriever.

This answers:

> "Did the retriever retrieve the right information?"

### Generator

Implemented in:

```text
evals/eval_generator.py
```

Metrics:

- Faithfulness
- Answer Relevancy

The generator is deliberately tested using **golden context** rather than live retrieved context.

This is important because it isolates the generator:

```text
Golden context
      │
      ▼
Generator
      │
      ▼
Answer
      │
      ├── Faithfulness
      └── Answer Relevancy
```

If generator faithfulness is poor here, the problem is likely in the generator/prompt rather than retrieval.

---

# 5. End-to-End RAG Pipeline Evaluation

Implemented in:

```text
evals/eval_rag_pipeline.py
```

This evaluates the complete:

```text
Query
  ↓
Retriever
  ↓
Reranker
  ↓
Generator
  ↓
Answer
```

Metrics:

- Contextual Relevancy
- Faithfulness
- Answer Relevancy

This is the classic RAG evaluation triad.

### Contextual Relevancy

Measures whether the retrieved context is relevant to the question.

A poor score can indicate:

- bad chunking
- bad top-k
- poor embeddings
- weak retrieval
- irrelevant retrieved chunks

### Faithfulness

Measures whether the generated answer is grounded in the retrieved context.

Conceptually:

```text
Faithfulness ≈ supported claims / total claims
```

A lower score indicates more unsupported or hallucinated claims.

### Answer Relevancy

Measures whether the final answer actually addresses the user's question.

Conceptually:

```text
Question
   ↓
Generated Answer
   ↓
Is the answer relevant to the question?
```

---

# 6. Application-Level Evaluation

Implemented in:

```text
evals/eval_application.py
```

This evaluates the system from the user's perspective rather than focusing only on retrieval internals.

Metrics:

### Correctness

Checks whether factual claims in the generated answer agree with the expected answer.

It intentionally does not punish an answer simply for being shorter.

### Completeness

Checks whether the answer covers the important points expected by the reference answer.

This separates:

```text
Correctness = "Are the claims correct?"
Completeness = "Did the answer cover what it needed to cover?"
```

### Style

Evaluates whether the answer follows the desired teaching style:

- clear
- conversational
- intuitive
- explanatory
- not unnecessarily robotic
- technical terms explained when needed

These are implemented using DeepEval `GEval` with custom rubrics.

---

# 7. Safety Evaluation

Implemented in:

```text
evals/eval_safety.py
```

Safety is treated differently from normal quality metrics.

A small quality regression may deserve human review.

A security or privacy regression may need to **block promotion**.

The project evaluates:

## Scope adherence

Checks whether the assistant follows the intended scope of the application.

## Protected-information leakage

Checks whether the assistant exposes:

- hidden system instructions
- internal instructions
- private operating rules
- raw retrieved context
- substantial protected course content

## PII leakage

Uses DeepEval's `PIILeakageMetric`.

The goal is to prevent the assistant from exposing sensitive personally identifiable information.

## Toxicity

Uses DeepEval's `ToxicityMetric`.

Unlike most quality metrics:

```text
Higher quality score = better
Higher toxicity score = worse
```

Therefore the metric registry treats toxicity as a **lower-is-better** metric.

---

# 8. Operational Evaluation

Implemented primarily in:

```text
evals/eval_ops.py
```

The project treats production readiness as more than model quality.

It measures:

## Latency

Includes:

- end-to-end mean
- p50
- p95
- p99
- retrieval latency
- generation latency
- time-to-first-token (TTFT)

The main regression signal is:

```text
ops.latency.e2e_p95_ms
```

P95 is useful because averages can hide slow requests.

## Cost

The cost evaluator measures actual token usage and estimates:

```text
input cost
+
cached input cost
+
output cost
=
cost per query
```

It also projects:

```text
daily cost
monthly cost
```

and checks the configured per-query budget.

## Reliability

Measures:

```text
success rate
error rate
retry rate
```

The benchmark includes retry handling with exponential backoff.

---

# 9. Golden Datasets

The project uses fixed JSON datasets so evaluation is repeatable.

Located under:

```text
goldens/
```

Current datasets contain 15 cases each:

```text
correctness_goldens.json
faithfulness_dataset.json
leakage_goldens.json
retriever_deepeval_goldens.json
retriever_goldens.json
scope_goldens.json
toxicity_goldens.json
```

The important principle is:

> The application can change; the evaluation dataset should remain stable.

This allows a candidate pipeline to be compared against a known baseline using the same questions and evaluation cases.

---

# 10. Evaluation Harness

Implemented in:

```text
evals/harness.py
```

The harness normalizes DeepEval results into a consistent summary:

```text
metric
├── n
├── pass_rate
├── avg_score
├── min_score
└── max_score
```

The project keeps metrics separate instead of pooling everything into one number.

For example:

```text
faithfulness = 0.96
answer_relevancy = 0.98
contextual_recall = 0.96
```

is much more useful than:

```text
overall_score = 0.96
```

because the individual metrics tell you **where** the system changed.

---

# 11. Metric Registry — The Core of Regression Testing

Implemented in:

```text
evals/metric_registry.py
```

This file is the conceptual heart of the regression system.

Every metric receives three pieces of policy:

```text
direction
kind
tolerance
```

## Direction

Defines whether higher or lower is better.

Examples:

```text
Faithfulness        → higher is better
Recall              → higher is better
Correctness         → higher is better
Latency             → lower is better
Cost                → lower is better
Toxicity            → lower is better
```

## Kind

There are three categories.

### Gate

A gate protects a critical requirement.

```text
gate regression
      ↓
BLOCK
```

Safety metrics are treated as gates.

### Guardrail

A guardrail is a softer production-quality constraint.

```text
guardrail regression
      ↓
REVIEW
```

The change is not automatically rejected, but it requires a human decision.

### Info

Informational metrics are tracked but never affect the verdict.

Examples:

- p50 latency
- p99 latency
- token counts
- sample counts
- secondary operational values

---

# 12. Tolerance and Noise Handling

LLM evaluations are not perfectly deterministic.

Even when the code does not change, LLM-judge scores can move slightly between runs.

The project therefore uses tolerances.

The comparison logic is:

```text
tolerance =
    max(
        absolute_tolerance,
        relative_tolerance × |baseline|
    )
```

A worsening metric inside this tolerance is classified as:

```text
flat
```

rather than:

```text
regressed
```

This prevents small evaluation noise from breaking regression testing.

Configured examples:

```text
Safety average-score gate:
    absolute tolerance = 0.02

Quality average-score guardrail:
    absolute tolerance = 0.05

Latency:
    relative tolerance = 25%

Cost:
    relative tolerance = 15%
```

The values live in:

```text
evals/metric_registry.py
```

and are intended to be tuned based on the observed noise floor of the evaluation environment.

---

# 13. Why Average Score Instead of Pass Rate?

For judge-based metrics, the regression system primarily uses:

```text
avg_score
```

rather than:

```text
pass_rate
```

Pass rate is highly dependent on the chosen threshold.

For example, if the threshold is:

```text
0.70
```

a tiny score movement around `0.70` can cause many test cases to flip from pass to fail.

The average score provides a smoother regression signal.

The trade-off is that an average can hide a single severe failure. That is why safety is separately designed as a hard gate and should be monitored at the individual-case level as well.

---

# 14. Snapshot-Based Regression Testing

Implemented in:

```text
evals/run_suite.py
```

A snapshot contains:

```json
{
  "metadata": {
    "created_at": "...",
    "git_sha": "...",
    "prompt_hash": "...",
    "label": "...",
    "suite_seconds": 123.4,
    "full": false,
    "n_metrics": 22
  },
  "metrics": {
    "retriever.contextual_recall.avg_score": 0.96,
    "generator.faithfulness.avg_score": 0.95
  }
}
```

This makes an evaluation run an immutable measurement of a particular pipeline state.

The metadata records:

- timestamp
- Git commit
- generator prompt hash
- human-readable label
- suite execution time
- number of persisted metrics

---

# 15. Baseline vs Candidate Workflow

The intended workflow is:

```text
                  ┌──────────────────┐
                  │ Known-good code  │
                  └────────┬─────────┘
                           │
                           ▼
                  run evaluation suite
                           │
                           ▼
                  baseline.json
                           │
                           │
                 change RAG system
                           │
                           ▼
                  run evaluation suite
                           │
                           ▼
                  candidate.json
                           │
                           ▼
                    compare.py
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
            PASS         REVIEW        FAIL
```

---

# 16. Running the Evaluation Suite

## Install `uv`

If `uv` is not installed, install it using the official `uv` installation instructions.

Then from the project root:

```bash
uv sync
```

The project requires Python 3.11 or newer.

---

# 17. Environment Variables

Create a local `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

Do not commit `.env`.

The repository's `.gitignore` already excludes it.

The project uses OpenAI for:

- embeddings
- generation
- DeepEval LLM judging

---

# 18. Build the Vector Store

The first retriever initialization creates the Chroma store if it does not already exist.

The project uses:

```text
Embedding model:
text-embedding-3-large
```

The retriever chunks transcript data using:

```text
chunk_size    = 1000
chunk_overlap = 150
```

The generated vector store is ignored by Git:

```text
chroma_store/
```

so it can be rebuilt locally.

---

# 19. Run the RAG Application

The UI is implemented in:

```text
src/app.py
```

The application is a Streamlit chat interface.

The sidebar allows you to change:

```text
fetch_k
top_k
```

`fetch_k` controls how many candidates are retrieved before reranking.

`top_k` controls how many chunks survive reranking and are passed to the generator.

If Streamlit is not already installed in your environment, add it:

```bash
uv add streamlit
```

Then run:

```bash
uv run streamlit run src/app.py
```

---

# 20. Run Individual Evaluations

Retriever:

```bash
python -m evals.eval_retriever
```

Generator:

```bash
python -m evals.eval_generator
```

End-to-end RAG pipeline:

```bash
python -m evals.eval_rag_pipeline
```

Application quality:

```bash
python -m evals.eval_application
```

The repository also contains dedicated safety and operational evaluation modules.

---

# 21. Create a Baseline

A known-good version should be evaluated first.

```bash
python -m evals.run_suite --baseline --label "current production"
```

This creates:

```text
baselines/baseline.json
```

For a full snapshot containing informational metrics:

```bash
python -m evals.run_suite --baseline --full --label "current production"
```

---

# 22. Evaluate a Candidate

After changing the RAG system:

```bash
python -m evals.run_suite --label "chunk=500 overlap=100"
```

This writes:

```text
baselines/candidate.json
```

You can also specify a custom output path:

```bash
python -m evals.run_suite --out baselines/experiment-001.json --label "experiment 001"
```

---

# 23. Compare Baseline and Candidate

Run:

```bash
python -m evals.compare
```

To display informational metrics too:

```bash
python -m evals.compare --all
```

Or compare explicit snapshots:

```bash
python -m evals.compare \
  --baseline baselines/baseline.json \
  --candidate baselines/candidate.json \
  --all
```

---

# 24. Understanding the Verdict

The current comparison engine returns three high-level outcomes.

## PASS

```text
No gate blocked.
No guardrail regressed beyond tolerance.
```

The candidate is safe to promote from the regression engine's perspective.

## REVIEW

```text
At least one guardrail regressed beyond tolerance.
```

The system recommends human review.

## FAIL

```text
At least one gate regressed.
```

A critical requirement, typically safety-related, was violated.

The command also returns CI-friendly exit codes:

```text
PASS   = 0
FAIL   = 1
REVIEW = 2
```

---

# 25. Example Regression Logic

Suppose:

```text
baseline faithfulness = 0.95
candidate faithfulness = 0.93
```

If the allowed tolerance is:

```text
0.05
```

then:

```text
delta = -0.02
```

Since:

```text
0.02 <= 0.05
```

the metric is:

```text
flat
```

Now suppose:

```text
baseline = 0.95
candidate = 0.85
```

Then:

```text
delta = -0.10
```

and:

```text
0.10 > 0.05
```

Therefore the metric is:

```text
regressed
```

If that metric is a guardrail:

```text
REVIEW
```

If it is a gate:

```text
FAIL
```

---

# 26. Project Structure

```text
rag-eval-deepeval/
│
├── data/
│   ├── YT Sandbox LLM Evals Session 1.vtt
│   ├── ...
│   └── YT Sandbox LLM Evals Session 8.vtt
│
├── goldens/
│   ├── correctness_goldens.json
│   ├── faithfulness_dataset.json
│   ├── leakage_goldens.json
│   ├── retriever_deepeval_goldens.json
│   ├── retriever_goldens.json
│   ├── scope_goldens.json
│   └── toxicity_goldens.json
│
├── src/
│   ├── app.py
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── reranker.py
│   └── retriever.py
│
├── evals/
│   ├── compare.py
│   ├── eval_application.py
│   ├── eval_cost.py
│   ├── eval_generator.py
│   ├── eval_latency.py
│   ├── eval_leakage.py
│   ├── eval_ops.py
│   ├── eval_rag_pipeline.py
│   ├── eval_reliability.py
│   ├── eval_retriever.py
│   ├── eval_retriever_with_reranker.py
│   ├── eval_safety.py
│   ├── eval_scope_safety.py
│   ├── eval_toxicity.py
│   ├── harness.py
│   ├── metric_registry.py
│   └── run_suite.py
│
├── resources/
│   └── deepeval_intro.py
│
├── baselines/
│   ├── baseline.json
│   └── candidate.json
│
├── chroma_store/
│   └── generated locally
│
├── .env
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 27. RAG Pipeline Implementation

The main pipeline is:

```python
class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        self.retriever = RerankingRetriever(
            fetch_k=fetch_k,
            top_k=top_k
        )

    def invoke(self, query):
        docs = self.retriever.invoke(query)

        context = [
            doc.page_content
            for doc in docs
        ]

        answer = generate(query, context)

        return {
            "query": query,
            "context": context,
            "answer": answer,
        }
```

The important design decision is that the pipeline returns all three parts:

```text
query
context
answer
```

This makes evaluation possible without reconstructing intermediate state.

---

# 28. Retriever + Reranker Design

The retriever uses:

```text
OpenAI text-embedding-3-large
        ↓
Chroma similarity search
        ↓
fetch_k candidates
        ↓
CrossEncoder reranking
        ↓
top_k final chunks
```

The reranker is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This gives the system a two-stage retrieval architecture:

```text
Stage 1:
Fast semantic retrieval

Stage 2:
More expensive cross-encoder ranking
```

This is useful for experimenting with the trade-off between:

```text
recall
precision
latency
cost
```

---

# 29. Generator Design

The generator uses:

```text
gpt-4o-mini
temperature = 0
```

The prompt is explicitly faithfulness-first.

The generator is instructed to:

- use only retrieved context
- avoid unsupported claims
- answer all relevant parts of the question
- explain concepts clearly
- maintain a teaching tone
- avoid toxic behavior
- avoid exposing system instructions
- avoid dumping protected course content
- abstain when the context does not contain enough information

The project therefore evaluates both:

```text
Prompt behavior
+
Generated answer behavior
```

---

# 30. Why Component-Level + Pipeline-Level + Application-Level Evaluation?

This is one of the most important design decisions in the project.

Imagine:

```text
Retriever score ↓
Generator score ↑
Final answer score ≈ same
```

An end-to-end evaluation alone might say:

```text
Everything looks fine.
```

But component-level evaluation reveals:

```text
Retriever is getting worse.
Generator is compensating for it.
```

Another scenario:

```text
Retriever ↑
Generator ↓
Application correctness ↓
```

Now the problem is likely downstream.

The evaluation architecture makes debugging much faster:

```text
Component level
      ↓
Find the failing subsystem

Pipeline level
      ↓
Measure RAG interaction

Application level
      ↓
Measure user-facing quality

Safety level
      ↓
Protect critical behavior

Operations
      ↓
Measure production feasibility
```

---

# 31. Reproducibility

The project records:

```text
Git SHA
Prompt hash
Timestamp
Human-readable experiment label
Suite duration
Metric count
```

The prompt hash is particularly useful.

A prompt change can significantly alter an LLM application's behavior even if the Python code barely changes.

Therefore:

```text
prompt change
      ↓
new prompt hash
      ↓
new evaluation snapshot
      ↓
baseline/candidate comparison
```

This makes silent prompt regressions easier to detect.

---

# 32. Production-Style Regression Testing

A practical workflow for development is:

```text
1. Create baseline
2. Change one thing
3. Run evaluation suite
4. Compare candidate to baseline
5. Inspect regressions
6. Decide whether to promote
7. If promoted, bless candidate as the next baseline
```

For example:

```text
Baseline:
chunk_size=1000
chunk_overlap=150
top_k=5

Experiment:
chunk_size=500
chunk_overlap=100
top_k=5
```

Then compare:

```text
retrieval
generation
pipeline
application
safety
cost
latency
reliability
```

This is much safer than optimizing only one metric.

---

# 33. CI/CD Integration

The comparison command is designed to work as a CI gate.

A typical CI flow can be:

```text
Pull Request
     │
     ▼
Install dependencies
     │
     ▼
Run evaluation suite
     │
     ▼
Generate candidate.json
     │
     ▼
Compare against baseline
     │
     ├── PASS ───────► merge/deploy
     │
     ├── REVIEW ─────► human review
     │
     └── FAIL ───────► block deployment
```

Because the comparator exposes process exit codes, it can be called directly from a CI job.

For example:

```bash
python -m evals.compare
```

A future GitHub Actions workflow can use the exit code to prevent deployment when a critical gate fails.

---

# 34. Important Design Principles

## 1. Evaluate components independently

Do not blame the generator for bad retrieval.

## 2. Evaluate the whole pipeline

Component scores do not always predict end-to-end behavior.

## 3. Keep golden datasets fixed

Otherwise the benchmark itself changes with the application.

## 4. Do not collapse every metric into one score

A single aggregate score can hide important failures.

## 5. Use tolerances

LLM evaluation has natural variance.

## 6. Treat safety differently

Security/privacy failures should not be treated like small quality fluctuations.

## 7. Track operational metrics

A high-quality RAG system that is too slow or too expensive may still be unusable.

## 8. Record provenance

Every snapshot should tell you which code and prompt produced it.

---

# 35. Limitations

This project is designed as an evaluation framework rather than a fully production-hardened RAG service.

Important limitations include:

- LLM-as-a-judge metrics are not perfectly deterministic.
- Evaluation quality depends on the quality of the golden datasets.
- Safety metrics can still miss novel attack patterns.
- Cost calculations depend on configured provider pricing.
- Latency measurements can vary because external model APIs are shared systems.
- The current vector store is local Chroma.
- The Streamlit UI is a development/demo interface.
- `baselines/` snapshots should be managed deliberately rather than regenerated blindly.
- A real production deployment should add authentication, observability, secrets management, rate limiting, and deployment infrastructure as required.

---

# 36. Future Improvements

Possible next steps:

### Evaluation

- Add more domain-specific goldens.
- Add human evaluation.
- Add citation correctness.
- Add hallucination-specific adversarial cases.
- Add multilingual evaluation.
- Add conversational/multi-turn RAG evaluation.

### Retrieval

- Experiment with multiple embedding models.
- Compare different chunking strategies.
- Add metadata filtering.
- Add hybrid BM25 + vector retrieval.
- Tune reranker models.
- Automatically optimize `fetch_k` and `top_k`.

### Regression

- Store historical evaluation snapshots.
- Generate HTML evaluation reports.
- Add automatic metric trend charts.
- Add GitHub PR comments.
- Add CI/CD deployment blocking.
- Add experiment tracking.

### Safety

- Add adversarial prompt suites.
- Add prompt injection evaluation.
- Add indirect prompt injection tests.
- Add sensitive-data canaries.
- Add red-team regression datasets.

### Operations

- Add tracing.
- Add distributed metrics.
- Add production traffic sampling.
- Add token/cost dashboards.
- Add real-time latency monitoring.

---

# 37. Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Application and evaluation code |
| DeepEval | LLM/RAG evaluation |
| LangChain | RAG orchestration |
| OpenAI | Embeddings, generation and judge model |
| Chroma | Vector database |
| Sentence Transformers | Cross-encoder reranking |
| Streamlit | RAG chat UI |
| pytest | Testing dependency |
| uv | Python environment and dependency management |
| JSON | Golden datasets and evaluation snapshots |

---

# 38. DeepEval Metrics Used

The project uses DeepEval's RAG-oriented metrics:

```text
Retriever
├── Contextual Precision
└── Contextual Recall

Generator
├── Faithfulness
└── Answer Relevancy

End-to-end RAG
└── Contextual Relevancy
```

It also uses:

```text
GEval
PIILeakageMetric
ToxicityMetric
```

for application and safety requirements.

DeepEval's documentation describes contextual precision as evaluating whether relevant retrieved nodes are ranked higher, contextual recall as whether the retrieval context captures information needed for the expected answer, contextual relevancy as the relevance of retrieved context to the input, faithfulness as alignment between the generated answer and retrieval context, and answer relevancy as alignment between the answer and user input.

---

# 39. Key Takeaway

This project is not simply a RAG chatbot.

It demonstrates a **RAG evaluation and release-safety system**:

```text
             RAG APPLICATION
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   RETRIEVAL                 GENERATION
        │                       │
        └───────────┬───────────┘
                    ▼
             END-TO-END RAG
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   APPLICATION    SAFETY       OPS
     QUALITY                   COST
                              LATENCY
                            RELIABILITY
                    │
                    ▼
             EVALUATION SNAPSHOT
                    │
                    ▼
          BASELINE vs CANDIDATE
                    │
                    ▼
             REGRESSION TEST
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        PASS      REVIEW      FAIL
```

The key engineering principle is:

> **Every RAG change should be treated like a software change: measure it, compare it against a known-good baseline, understand the trade-offs, and only then promote it.**

---

## References

- DeepEval RAG evaluation: https://deepeval.com/guides/guides-rag-evaluation
- DeepEval metrics overview: https://deepeval.com/docs/metrics-introduction
- DeepEval RAG quickstart: https://deepeval.com/docs/getting-started-rag
- DeepEval contextual precision: https://deepeval.com/docs/metrics-contextual-precision
- DeepEval contextual recall: https://deepeval.com/docs/metrics-contextual-recall
- DeepEval contextual relevancy: https://deepeval.com/docs/metrics-contextual-relevancy
- DeepEval faithfulness: https://deepeval.com/docs/metrics-faithfulness
- DeepEval answer relevancy: https://deepeval.com/docs/metrics-answer-relevancy
- DeepEval G-Eval: https://deepeval.com/docs/metrics-llm-evals
