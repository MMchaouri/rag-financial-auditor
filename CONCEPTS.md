# Financial Compliance Auditor — Concept Guide

Every concept in this project explained at three levels:
- **Like I'm 5** — pure analogy, no jargon
- **Intuition** — the mental model an engineer builds
- **Graduate level** — precise technical framing

Work through this top-to-bottom once. Revisit each section as you implement the matching code.

---

## Table of Contents

1. [RAG — Retrieval-Augmented Generation](#1-rag--retrieval-augmented-generation)
2. [Embeddings](#2-embeddings)
3. [Vector Database / ChromaDB](#3-vector-database--chromadb)
4. [Chunking Strategy](#4-chunking-strategy)
5. [Sentence Transformers](#5-sentence-transformers)
6. [LLM (Large Language Model)](#6-llm-large-language-model)
7. [Prompt Engineering](#7-prompt-engineering)
8. [Source Grounding & Hallucination Prevention](#8-source-grounding--hallucination-prevention)
9. [LangChain](#9-langchain)
10. [FastAPI](#10-fastapi)
11. [Compliance Audit Logic](#11-compliance-audit-logic)
12. [Structured Outputs](#12-structured-outputs)
13. [Evaluation Framework](#13-evaluation-framework)
14. [Information Retrieval (IR)](#14-information-retrieval-ir)
15. [Docker (Optional Stretch)](#15-docker-optional-stretch)

---

## 1. RAG — Retrieval-Augmented Generation

### Like I'm 5
Imagine you have a really smart friend who knows a lot of general stuff, but has never read your school's rulebook.
You ask them "Did I break rule #47?" — they'd just guess.
Now imagine before answering, your friend is allowed to go grab the rulebook, flip to the relevant page, read it, and *then* answer.
That's RAG. The friend is the LLM. The rulebook is your PDF. "Going to grab it" is retrieval.

### Intuition
A pure LLM only knows what was baked into its weights at training time. It cannot know your specific 10-K filing.
RAG splits the problem in two:
- **Retrieval**: find the relevant pages from your document given a question
- **Generation**: give those pages to the LLM as context and let it reason over them

The LLM acts as a reasoning engine, not a knowledge store.
This matters enormously for compliance: you need *verifiable, document-backed* answers, not hallucinated ones.

### Graduate Level
RAG is a fusion architecture: a parametric model (the LLM, which stores knowledge in weights) augmented with a non-parametric retrieval component (the vector store).
Formally: given a query `q`, retrieve top-k documents `D = {d1...dk}` from corpus `C`, then condition generation on `P(answer | q, D)` rather than `P(answer | q)` alone.
This decouples the knowledge base from the model weights, enabling zero-shot adaptation to new documents without fine-tuning.
The original paper is Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Facebook AI Research).

---

## 2. Embeddings

### Like I'm 5
Words are like people. If you put all the people at a party on a map — people who like the same things stand close to each other, people who hate each other stand far apart.
An embedding is that map. It turns a sentence into a spot on the map (a list of numbers).
"Revenue recognition" and "how we count money" end up close on the map, so the computer knows they're related.

### Intuition
An embedding is a fixed-length vector (e.g., 768 floats) that encodes the *meaning* of text, not just its characters.
Two semantically similar sentences produce vectors that are geometrically close (low cosine distance).
This lets us search by meaning rather than keyword. A query "Is liquidity discussed?" can find a chunk that says "cash flow management" — even though none of those words overlap.

### Graduate Level
Embeddings are learned via contrastive training objectives (e.g., SimCSE, MNRL).
The model is trained so that `cos_sim(encode(A), encode(B))` is high when A and B are semantically equivalent (positive pair) and low for random negatives.
The resulting vector space is an approximate metric space over semantic similarity.
For retrieval, we care about the nearest-neighbor problem: given query embedding `q`, find documents `d_i` maximizing `cos_sim(q, d_i)`.
This is solved efficiently with approximate nearest neighbor (ANN) algorithms (HNSW, IVF) inside ChromaDB.

---

## 3. Vector Database / ChromaDB

### Like I'm 5
You have thousands of sticky notes from a big book. You want to find "all notes about money problems."
If you line them up and read each one, that takes forever. But what if each sticky note had a tiny GPS dot on it — notes about money cluster in one corner of your room. You just go to that corner.
ChromaDB is the room. The GPS dots are embeddings. Finding the corner is vector search.

### Intuition
A vector database stores embeddings alongside metadata and supports fast semantic search.
ChromaDB is an embedded database (runs in-process, no server needed) which makes it perfect for prototyping.
When you run a query, ChromaDB converts your query to an embedding then returns the top-k stored chunks whose embeddings are closest — this is cosine similarity search.
Metadata filtering (e.g., "only search year=2023") narrows the search before or after the ANN step.

### Graduate Level
ChromaDB uses HNSW (Hierarchical Navigable Small World) graphs under the hood for ANN search.
HNSW provides `O(log n)` query complexity with tunable recall/speed tradeoffs via `ef` and `M` parameters.
The schema in this project stores: `doc_name`, `page_number`, `section_title`, `chunk_id`, `filing_year`, `company_name`.
These metadata fields enable hybrid filtering: SQL-style predicate pushdown combined with ANN — so you can say "find semantically similar chunks, but only from Apple's 2023 10-K."
This is critical for compliance: you never want evidence from a different company's filing contaminating your audit.

---

## 4. Chunking Strategy

### Like I'm 5
Imagine you eat a whole pizza in one bite — impossible, you'd choke.
So you cut it into slices. But if the slices are too small (one pepperoni each), you lose the flavor.
Chunking is cutting a giant PDF into "just-right" pieces so the AI can digest them.

### Intuition
LLMs have a context window (max tokens they can see at once). You can't feed a 300-page 10-K to an LLM directly.
Chunks need to be:
- **Big enough** to contain complete thoughts (a sentence about revenue recognition should stay whole)
- **Small enough** to be retrieved precisely (a 50-page chunk is useless as "evidence")

Common strategies:
- **Fixed-size with overlap**: 512 tokens, 50 token overlap between consecutive chunks. Simple, works well.
- **Semantic chunking**: split on semantic boundaries (paragraph breaks, section headers). Better for structured docs like 10-Ks.
- **Recursive character splitting**: LangChain's default, tries `\n\n`, then `\n`, then ` ` as split points.

For 10-Ks, semantic chunking on section headers is the right call — the SEC requires specific sections (MD&A, Risk Factors), so split there first, then sub-chunk.

### Graduate Level
Chunk size vs. overlap is a retrieval precision/recall tradeoff.
Larger chunks → higher recall (more context per hit) but lower precision (noisy retrieved context degrades LLM reasoning).
Overlap prevents important information from being split across chunk boundaries — the key invariant is that every sentence appears fully in at least one chunk.
The optimal chunk size is task-dependent; for compliance auditing, ~512-768 tokens with 10-15% overlap is a reasonable default.
Advanced: "parent-child chunking" (LlamaIndex SmallToLarge) stores small chunks for retrieval precision but expands to parent chunks for LLM context — reduces hallucination risk.

---

## 5. Sentence Transformers

### Like I'm 5
You know how a translator takes French words and turns them into English? Sentence Transformers are translators that take sentences and turn them into numbers — but in a way that similar sentences become similar numbers.

### Intuition
Sentence Transformers (the library and model family by UKPLab) are BERT-based models fine-tuned specifically for producing good sentence embeddings.
The key model for this project: `all-MiniLM-L6-v2` — fast, small, 384-dimensional embeddings.
For higher quality: `all-mpnet-base-v2` — 768-dim, slower, better semantic capture.
For financial domain: `BAAI/bge-base-en-v1.5` — trained on broader corpus, strong on dense retrieval benchmarks (BEIR).

### Graduate Level
Sentence Transformers use a Siamese network architecture with mean-pooling over BERT token embeddings to produce fixed-length sentence representations.
Fine-tuning uses the Multiple Negatives Ranking Loss (MNRL): for a batch of (anchor, positive) pairs, every other positive becomes a hard negative — this is computationally efficient and produces strong encoders.
For financial compliance, domain-adapted models (e.g., FinBERT-based encoders) may improve retrieval on accounting terminology, but the performance gap shrinks when chunk quality is high.

---

## 6. LLM (Large Language Model)

### Like I'm 5
Think of the LLM as the world's most well-read intern. They've read billions of pages of text and are really good at understanding and writing language. But they only know what they read before their "graduation" — nothing after. And they can make stuff up confidently (that's the dangerous part). Your job as the architect is to make sure they only answer based on documents you hand them.

### Intuition
The LLM in this project is the reasoning engine — it receives:
1. A compliance requirement (e.g., "Is a risk factors section present and substantive?")
2. Retrieved chunks from the filing
3. A strict prompt instructing it to only use provided context

Its job: analyze context, produce a structured verdict (Pass/Fail + confidence + quoted evidence).
Open-source options: `Mistral-7B-Instruct`, `Llama-3-8B-Instruct`, `Phi-3-mini`. Run locally via `ollama` or `llama.cpp`. Use `HuggingFace Hub` for hosted inference.

### Graduate Level
LLMs are autoregressive transformers trained on next-token prediction at scale (scaling laws: Chinchilla optimal compute allocation).
For this project, instruction-tuned variants (RLHF or DPO-aligned) are required — base models don't follow structured output instructions reliably.
The key architectural concern is the context window: most 7B models handle 4K-8K tokens; some (Mistral, Llama-3) handle 32K.
For compliance auditing, context length matters because some retrieved evidence is long. Budget: ~2K tokens for retrieved context, ~500 for the system prompt and compliance requirement, ~500 for the structured output schema, with margin.
Quantized models (GGUF 4-bit) run on consumer hardware (16GB RAM) with acceptable quality degradation.

---

## 7. Prompt Engineering

### Like I'm 5
If you tell a genie "give me something to eat," you might get a rock (technically edible). But if you say "give me a warm pepperoni pizza from Dominos, 12 inches, thin crust" — you get what you want. Prompts are instructions to the AI. Better instructions = better results.

### Intuition
For compliance auditing, prompts must be:
- **Specific about format**: tell the model to return JSON with exact keys
- **Grounding instructions**: "Only use the provided context. If the context doesn't support a finding, say 'insufficient evidence'."
- **Persona**: "You are a financial compliance auditor reviewing SEC filings."
- **Output constraints**: "Confidence must be between 0.0 and 1.0. Evidence must be a verbatim quote."

The prompt is the primary lever you control. A bad prompt with a good model gives bad results. A good prompt with a weaker model often beats a bad prompt with a stronger model.

### Graduate Level
Prompt engineering for structured, grounded outputs involves several techniques:
- **Chain-of-Thought (CoT)**: instruct the model to reason step-by-step before producing the final JSON. Improves accuracy on multi-step compliance checks.
- **Few-shot examples**: provide 1-2 example (requirement → evidence → verdict) triples in the system prompt. Dramatically improves output format compliance.
- **Constrained generation**: if using `llama.cpp` or `vLLM`, use grammar-constrained decoding (GBNF/LMQL) to enforce valid JSON structure at the token sampling level — eliminates malformed output entirely.
- **Grounding instruction**: explicit "do not introduce information not present in the provided context" is empirically shown to reduce hallucination by 40-60% in RAG settings (DeepMind, 2023).

---

## 8. Source Grounding & Hallucination Prevention

### Like I'm 5
Imagine a friend tells you "the rule says you can't wear red shoes." But when you check the rulebook, it says nothing about shoes. Your friend made it up — that's a hallucination. Source grounding means the friend must point to the exact sentence in the book before they're allowed to say something is a rule.

### Intuition
Hallucination = the LLM generating confident-sounding claims not supported by the retrieved context.
In a compliance context, hallucinations are not just wrong — they're dangerous (false pass on a regulatory requirement).
Prevention mechanisms:
1. **Citation enforcement**: the prompt requires `"evidence"` to be a verbatim quote from the retrieved chunk. If the model can't find a quote, it must return `"status": "Insufficient Evidence"`.
2. **Confidence scoring**: the model outputs its own confidence (0-1). Low confidence = flag for human review.
3. **Post-hoc verification**: after the LLM produces an evidence quote, programmatically check that the quote exists in the retrieved chunks using fuzzy string matching. If it doesn't exist → hallucination detected → output overridden.
4. **Context-only prompting**: inject retrieved chunks verbatim. Don't paraphrase.

### Graduate Level
Hallucination in RAG systems is categorized as:
- **Intrinsic hallucination**: model contradicts its own retrieved context
- **Extrinsic hallucination**: model introduces information absent from context

Detection methods:
- **NLI-based verification**: use a Natural Language Inference model (e.g., `cross-encoder/nli-deberta-v3-base`) to check if the generated claim is entailed by the retrieved context. This is the programmatic grounding check layer.
- **SelfCheckGPT**: sample the model multiple times; claims that appear consistently across samples are more likely grounded.
- **Citation recall metric**: fraction of LLM claims that can be traced to a specific retrieved chunk passage.

This project implements the programmatic fuzzy-match check as the minimum viable grounding layer, with NLI as a stretch goal.

---

## 9. LangChain

### Like I'm 5
Building with AI is like building with Lego. LangChain is the Lego starter kit — it has pre-built pieces for "load a PDF," "split into chunks," "search a vector database," "call an AI." You assemble them instead of making every brick from scratch.

### Intuition
LangChain provides abstractions for the common patterns in LLM applications:
- `DocumentLoader` → reads PDFs, HTML, etc.
- `TextSplitter` → chunks documents
- `Embeddings` → wraps sentence-transformer or OpenAI embeddings
- `VectorStore` → wraps ChromaDB, FAISS, Pinecone, etc.
- `RetrievalQA` / `LCEL` chains → wires retrieval + LLM together
- `OutputParser` → parses LLM output into structured Python objects

In this project, LangChain handles the "plumbing." You write the compliance logic; LangChain handles PDF loading, chunk storage, retrieval, and LLM calls.

### Graduate Level
LangChain Expression Language (LCEL) is the modern API (replaces legacy chain classes).
It uses a pipe operator (`|`) to compose Runnables — composable, streaming-capable units.
Example pipeline:
```
retriever | format_docs | prompt | llm | output_parser
```
Each step is lazy and supports `.stream()`, `.batch()`, `.ainvoke()` natively.
For this project, LCEL chains are used for the retrieval-reasoning pipeline. Avoid the legacy `RetrievalQA` class — it has less control over prompt injection and context formatting.
Key LCEL concepts: `RunnableParallel` (run retrieval and formatting in parallel), `RunnablePassthrough` (pass original query through unchanged), `RunnableLambda` (wrap any function).

---

## 10. FastAPI

### Like I'm 5
Your AI compliance system is like a super-smart machine in a factory. FastAPI builds the buttons and windows on the outside of the machine so other people (or websites) can use it without knowing how the machine works inside. They press a button (send a request), the machine does stuff, and a result comes out.

### Intuition
FastAPI is a Python web framework for building HTTP APIs. It's fast (async by default), validates inputs automatically with Pydantic, and auto-generates OpenAPI docs (Swagger UI).
In this project:
- `POST /upload` — accepts a PDF, runs the ingestion pipeline, stores chunks in ChromaDB
- `POST /audit` — runs the compliance audit for a specific company/document, returns a report ID
- `GET /report/{id}` — fetches the stored compliance scorecard JSON
- `GET /health` — liveness check (for Docker health checks and load balancers)

### Graduate Level
FastAPI uses Starlette (ASGI) under the hood with Uvicorn as the server.
All endpoints are `async def` to avoid blocking the event loop during I/O-heavy operations (PDF loading, ChromaDB queries, LLM calls).
Pydantic V2 models define request/response schemas with full type validation, serialization, and documentation generation automatically.
For production: add `BackgroundTasks` to run ingestion asynchronously (upload returns immediately, ingestion runs in background), add Redis for job status tracking, add authentication via `fastapi-users` or OAuth2 bearer tokens.
File upload uses `UploadFile` + `SpooledTemporaryFile` — files are kept in memory up to a threshold, then spooled to disk. For large 10-Ks (up to 200MB), configure `max_upload_size`.

---

## 11. Compliance Audit Logic

### Like I'm 5
The SEC (the financial police) has a checklist of things every big company must say in their annual report. Our AI goes through that checklist — for each item, it searches the report, reads the relevant part, and says "yes, the company talked about this" or "no, it didn't — FAIL."

### Intuition
SEC 10-K filings have required sections defined by Regulation S-K. Our compliance checklist maps to these requirements:

| Check | Required By |
|---|---|
| Risk Factors section | Item 1A |
| MD&A (Management Discussion & Analysis) | Item 7 |
| Quantitative disclosures about market risk | Item 7A |
| Financial statements + auditor info | Item 8 |
| Legal proceedings | Item 3 |
| Liquidity and capital resources | Part of Item 7 |
| Revenue recognition policy | ASC 606 disclosure |

For each check, the system queries ChromaDB with a targeted query (e.g., "risk factors disclosed by management"), retrieves top-k chunks, and prompts the LLM to rule Pass/Fail.

### Graduate Level
Compliance requirements map to Named Entity Recognition + semantic retrieval problems.
The challenge: "is Risk Factors present" is easy (section title lookup), but "is the revenue recognition discussion substantive" requires qualitative reasoning — this is where LLM reasoning is genuinely valuable.
Pass/Fail criteria are defined in structured `ComplianceRequirement` dataclasses:
- `query`: the retrieval query to run
- `section_indicator`: keywords/regex to pre-filter chunks
- `minimum_evidence_length`: reject single-word answers as insufficient
- `confidence_threshold`: minimum model confidence to accept a Pass

The audit loop iterates over requirements, retrieves, reasons, and aggregates into a scorecard with an overall compliance score (weighted average of individual check confidences × pass/fail binary).

---

## 12. Structured Outputs

### Like I'm 5
Instead of the AI writing a paragraph like a human, we tell it "only answer by filling in this form." Like a doctor who must fill in boxes — Name: ___, Diagnosis: ___, Confidence: ___. No rambling. No guessing outside the boxes.

### Intuition
LLMs naturally produce free text. For a compliance system, you need machine-parseable, consistent output.
Approach: define a Pydantic model for the output, serialize it to JSON schema, inject the schema into the prompt as an output format instruction, then parse the LLM's response.
LangChain provides `PydanticOutputParser` and `.with_structured_output()` for models that natively support JSON mode.

### Graduate Level
Structured output approaches by reliability tier:
1. **JSON mode** (OpenAI/Mistral hosted API): model guaranteed to return valid JSON. Can still produce wrong keys/types.
2. **Pydantic parser + retry**: parse LLM output → if parse fails → retry with error message injected. LangChain `OutputFixingParser` handles this.
3. **Grammar-constrained decoding** (llama.cpp / vLLM): enforce GBNF grammar at token sampling level — physically impossible to produce invalid JSON. Zero parse failures. Adds ~5% latency overhead.
4. **Function calling** (OpenAI, Groq, Ollama): model outputs a JSON dict matching a function signature — highest reliability for hosted models.

For this project: use `.with_structured_output(ComplianceFinding)` for local Ollama models, fall back to `PydanticOutputParser` with retry if structured output isn't supported.

---

## 13. Evaluation Framework

### Like I'm 5
After you build a robot, you test it. You give it problems you already know the answers to. If it gets 9/10 right, it scores 90%. We do the same for our AI — we check if it finds the right pages, quotes real text, and gets the right Pass/Fail answer.

### Intuition
Four things to measure:
1. **Retrieval precision@k** — of the k chunks retrieved, what fraction were actually relevant? (Human-labeled ground truth)
2. **Citation accuracy** — does the quoted `evidence` text actually appear verbatim in the retrieved chunks? (Programmatic check — fuzzy match)
3. **Grounding rate** — fraction of findings where evidence is non-empty and verifiable
4. **Hallucination rate** — fraction of findings where the LLM's claim is not entailed by the retrieved context (NLI check)
5. **Compliance classification accuracy** — if you have gold-labeled audits (Pass/Fail per requirement), what's the accuracy? F1?

### Graduate Level
Evaluation is the hardest part. For RAG systems, the RAGAS framework provides automated metrics:
- **Faithfulness**: measures if the answer is supported by context (NLI-based). Target: >0.85
- **Answer Relevancy**: cosine similarity between answer and original question. Target: >0.80
- **Context Precision**: are retrieved chunks actually useful? (LLM-as-judge). Target: >0.75
- **Context Recall**: are all needed chunks retrieved? Requires reference answers.

For compliance specifically, build a labeled test set: take 3-5 real 10-K filings, manually audit them against your checklist, record the ground-truth Pass/Fail for each requirement. This becomes your benchmark dataset.
Use `sklearn.metrics.classification_report` for precision/recall/F1 per compliance requirement. A confusion matrix shows which requirements are hardest for the system.

---

## 14. Information Retrieval (IR)

### Like I'm 5
Information Retrieval is the science of finding the right needle in a haystack of needles. Google is an IR system. So is your email search. Our system uses IR to find the right paragraphs in a 300-page PDF.

### Intuition
Two paradigms:
- **Sparse retrieval (BM25)**: keyword-based, fast, no vectors. TF-IDF variant. Works well for exact financial terms ("EBITDA", "going concern").
- **Dense retrieval (embeddings)**: semantic, handles paraphrase. Works well for conceptual queries ("liquidity concerns").

**Hybrid search** combines both: BM25 score + dense score, merged via Reciprocal Rank Fusion (RRF). Consistently outperforms either alone on financial documents because SEC filings mix formal jargon (sparse wins) with conceptual discussions (dense wins).

### Graduate Level
The dense retrieval pipeline is an Approximate Nearest Neighbor (ANN) problem in high-dimensional space (384 or 768 dims).
ChromaDB's HNSW index provides sub-millisecond queries on collections of 100K+ chunks.
Key IR metrics:
- **Recall@k**: fraction of relevant documents in top-k results
- **MRR (Mean Reciprocal Rank)**: where does the first relevant result appear?
- **NDCG (Normalized Discounted Cumulative Gain)**: quality-weighted ranking metric

Reranking (stretch goal): after initial retrieval (top-20 candidates), apply a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) that jointly encodes query + candidate and produces a relevance score. Reranker is slower but 10-15% more accurate than bi-encoder retrieval. Top-5 after reranking → LLM context.

---

## 15. Docker (Optional Stretch)

### Like I'm 5
Your app works on your computer but breaks on your friend's computer because they have different stuff installed. Docker puts your app in a lunchbox with everything it needs — food, utensils, napkin. The lunchbox works the same everywhere.

### Intuition
Docker packages your FastAPI app, Python dependencies, and ChromaDB storage into a container image.
Benefits for this project:
- Recruiter/interviewer can run `docker compose up` and see your demo in 30 seconds
- Shows production engineering awareness
- Enables multi-service setup: API container + (optional) separate Chroma server container

### Graduate Level
`Dockerfile` for a FastAPI app: multi-stage build (builder stage installs deps, runtime stage copies only what's needed).
`docker-compose.yml` defines:
- `api` service: FastAPI + uvicorn
- `chroma` service (optional): ChromaDB in client/server mode (replaces in-process mode)
- Volume mount for `data/` so PDFs persist across container restarts

Key considerations:
- Pin base image to a specific digest for reproducibility
- Use `.dockerignore` to exclude `venv/`, `__pycache__/`, `*.pdf`
- Set `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1`
- For M1/M2 Macs: use `--platform linux/amd64` only if deploying to x86 cloud; otherwise build native ARM

---

## How These Concepts Connect

```
PDF File
    ↓
[DocumentLoader] — LangChain reads the PDF
    ↓
[TextSplitter] — Chunking Strategy cuts it into pieces
    ↓
[SentenceTransformer] — Embeddings converts each chunk to a vector
    ↓
[ChromaDB] — Vector Database stores vectors + metadata
    ↓
                        ← Query (compliance requirement)
                               ↓
                        [SentenceTransformer] — embed the query too
                               ↓
                        [ChromaDB ANN Search] — find closest chunks
                               ↓
                        [LLM + Prompt Engineering] — reason over evidence
                               ↓
                        [Structured Output + Grounding Check]
                               ↓
                        [Compliance Scorecard JSON]
                               ↓
                        [FastAPI] — serves the result to clients
```

RAG = the arc from "Query" to "Scorecard."
Source Grounding = the check between LLM output and retrieved chunks.
LangChain = the code that wires all the arrows together.
Evaluation = the process of measuring whether every arrow works correctly.

---

## Resume Justification Summary

| Skill on Resume | Where It Appears in This Project |
|---|---|
| RAG | Core architecture: retrieval + generation pipeline |
| LangChain | Document loading, chunking, LCEL chain wiring |
| ChromaDB / Vector Databases | Embeddings storage, ANN retrieval, metadata filtering |
| Sentence Transformers | Embedding model for chunks and queries |
| LLM Applications | Compliance reasoning over retrieved evidence |
| FastAPI | REST API for upload, audit, report endpoints |
| Information Retrieval | Retrieval pipeline, precision@k evaluation |
| NLP | Chunking, tokenization, embedding, text extraction |
| Prompt Engineering | Structured output prompts, CoT, grounding instructions |
| AI Evaluation | RAGAS metrics, classification accuracy, hallucination rate |
| Source Grounding | Citation enforcement, fuzzy-match verification |
| Structured Outputs | Pydantic models, JSON schema injection, parsing |
