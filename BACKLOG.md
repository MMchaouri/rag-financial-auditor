# Project Backlog — RAG Financial Compliance Auditor

Each item below covers: what it is, why we chose it, and how else we could have done it.
Three levels: **Like I'm 5** → **Adult dev** → **Master's student**.

---

## DONE

---

### ✅ PDF Loader (`app/ingestion/loader.py`)

**What we built:** Read a PDF file page by page and turn it into a list of text blocks with page numbers.

#### Like I'm 5
Imagine you have a book and you rip out each page and write on a sticky note what page it came from. That's it. Now we have sticky notes instead of a book.

#### Adult dev
Used `pypdf` to extract raw text per page. Each page becomes a LangChain `Document` object — a standard wrapper that carries text plus a `metadata` dict (`page_number`, `source`). Downstream code doesn't need to care what file format it came from.

#### Master's student
`pypdf` is a pure-Python PDF parser targeting text-layer extraction. It works for machine-generated PDFs (SEC EDGAR filings are always machine-generated). Scanned PDFs would need OCR (`pytesseract` + `pdf2image`). We skip blank pages (`text.strip()`) to avoid poisoning the vector store with empty embeddings. Metadata is structural, not semantic — semantic enrichment (section headers) happens at chunking time.

**Alternatives considered:**
- `pdfminer.six` — more precise layout parsing, but heavier and slower for our use case
- `pymupdf (fitz)` — fastest, has C bindings, but GPL license; pypdf is MIT
- `unstructured` — handles mixed formats (PDF/HTML/DOCX) with layout awareness; overkill here, adds 15+ transitive deps

---

### ✅ Text Chunker (`app/ingestion/chunker.py`)

**What we built:** Take those sticky notes (pages) and cut them into smaller pieces so the AI can read them one at a time.

#### Like I'm 5
If a page is really long, the AI gets confused — it's like reading a whole chapter at once. So we cut each page into small paragraphs it can handle. We let them overlap a tiny bit so no important sentence gets cut in half.

#### Adult dev
`RecursiveCharacterTextSplitter` from LangChain cuts text by trying delimiters in order: `\n\n`, `\n`, ` `, `""`. Result is chunks of ~512 chars with 64-char overlap. Each chunk gets a `chunk_id` (`source_path + zero-padded index`) for traceability.

#### Master's student
512-char / 64-char overlap is a deliberate tradeoff: small enough to fit dense SEC prose into 384-dim MiniLM embedding space without semantic dilution; large enough to capture a full sentence + its surrounding context. The `RecursiveCharacterTextSplitter` is greedy-left — it prioritizes paragraph breaks, then line breaks, then word breaks, minimizing mid-sentence cuts. Overlap addresses the boundary problem: a compliance-relevant sentence near a chunk boundary appears in both adjacent chunks, ensuring at least one retrieval hit.

**Alternatives considered:**
- Fixed-size token chunking — more consistent but ignores sentence boundaries; common in OpenAI cookbook but lossy for legal prose
- Semantic chunking (embed → cluster → split at embedding-space discontinuities) — state-of-the-art but 5–10× slower at ingest; needs a second embedding pass
- Sentence-level chunking with `spacy` — cleaner splits but overkill; SEC prose has well-structured paragraphs already

---

### ✅ Embedder + Vector Store (`app/ingestion/embedder.py`)

**What we built:** Turn each text chunk into a list of numbers (a vector), then store all those vectors in a searchable database.

#### Like I'm 5
Imagine each chunk of text gets turned into a point on a huge invisible map. Chunks that talk about similar things land close together on the map. Later, when we ask a question, we drop our question on the map and find the closest points.

#### Adult dev
`HuggingFaceEmbeddings` wraps `all-MiniLM-L6-v2` locally — no API call, no cost per embed. Vectors stored in `ChromaDB` (on-disk, `./data/chroma`). Each chunk gets metadata: `company_name`, `filing_year`, `doc_name`, `doc_id` (UUID). The UUID is the primary key for retrieval filtering.

#### Master's student
`all-MiniLM-L6-v2` produces 384-dim L2-normalized vectors. It was distilled from a larger MPNet model via knowledge distillation specifically for symmetric semantic similarity tasks — which is exactly our use case (query and chunk are the same type of text). ChromaDB uses HNSW (Hierarchical Navigable Small World graph) for ANN search: O(log n) query time with configurable recall/speed tradeoff. `doc_id` UUID filter is a pre-filter that runs before HNSW traversal, scoping the search to one document's chunk graph.

**Alternatives considered:**
- `OpenAI text-embedding-3-small` — better quality, but $0.02/1M tokens and requires internet; we need offline/local
- `FAISS` — Meta's ANN library, faster at scale, but no built-in persistence or metadata filtering; would need a separate metadata store
- `Weaviate` / `Qdrant` — production-grade vector DBs with better multi-tenancy; overkill for a single-machine portfolio project
- `pgvector` — Postgres extension; excellent if you already have Postgres, but adds infra dependency

---

### ✅ Retriever (`app/retrieval/retriever.py`)

**What we built:** Given a question, find the 5 most relevant text chunks from one specific document.

#### Like I'm 5
Remember the map? Now we drop our question on the map and find the 5 closest sticky notes. But we only look inside sticky notes from *one company's filing* — not everyone's.

#### Adult dev
`vs.similarity_search(query, k=5, filter={"doc_id": doc_id})` — single call, returns top-k by cosine similarity, filtered to the document being audited. `doc_id` is the UUID assigned at upload time.

#### Master's student
The `filter` argument triggers ChromaDB's pre-filtering path: it intersects the candidate set with the metadata index before HNSW traversal, not after. This matters because post-filtering (retrieve top-k globally, then drop non-matching) degrades recall when one document's chunks are a small fraction of the collection. Pre-filter ensures the effective k is drawn from the right partition. `top_k=5` is configurable in `Settings` — SEC filings are dense, and 5 chunks × ~512 chars = ~2560 chars of context, which fits in mistral:7b-instruct's 4096-token context window with room for the prompt template.

**Alternatives considered:**
- MMR (Maximal Marginal Relevance) retrieval — reduces redundancy by penalizing near-duplicate chunks; useful if a requirement spans multiple sections that use similar language
- HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer, embed it, use that as the query vector; improves recall for vague queries; adds one LLM call per retrieval
- BM25 hybrid search — combine dense (embedding) + sparse (keyword TF-IDF) retrieval; better for exact legal phrase matching; ChromaDB doesn't support this natively (would need a second retriever + RRF fusion)

---

### ✅ Context Assembler (`app/retrieval/context.py`)

**What we built:** Take those 5 retrieved chunks and stitch them into one readable string to hand to the AI.

#### Like I'm 5
You found your 5 sticky notes. Now you tape them together on one big paper, write what page each came from, and hand it to the AI to read.

#### Adult dev
Formats each `Document` as `[Page N]\n{text}`, joins with `\n\n---\n\n` separator. This gives the LLM structural cues (page attribution) so it can cite evidence accurately. Returns a plain string — the prompt builder wraps it.

#### Master's student
The separator `---` acts as a structural delimiter that instruction-tuned models (mistral-instruct) reliably treat as a section boundary. Explicit `[Page N]` attribution enables source grounding — the LLM's quoted evidence can be fuzzy-matched back to the chunk's known page number. This is the first link in the hallucination-prevention chain: the LLM only sees real text with real page markers, so any citation it generates is checkable.

**Alternatives considered:**
- XML tags (`<chunk id="1" page="3">`) — more parseable but instruction-tuned models handle both equally; XML adds token overhead
- Numbered citations `[1]`, `[2]` with a footnote section — common in academic RAG papers (e.g., RARR); makes post-hoc citation extraction easier but complicates the prompt

---

## TODO

---

### ⬜ Task 7: LLM Client + Prompts (`app/llm/client.py`, `app/llm/prompts.py`)

**What we'll build:** Connect to the local Mistral AI model and write the question template we send it for each compliance check.

#### Like I'm 5
We have our sticky notes ready. Now we need to write a letter asking the AI: "Hey, based on these notes, does this company follow rule #5?" The letter has blanks we fill in each time.

#### Adult dev
`langchain-ollama` wraps the local Ollama HTTP API. The prompt template takes `{requirement}`, `{context}`, and asks for structured JSON output: `{status, confidence, evidence, page}`. `StrOutputParser` extracts the raw string; we parse JSON manually to avoid LangChain's JSON parser quirks.

#### Master's student
Mistral-7b-instruct uses `[INST]...[/INST]` prompt formatting. We use LangChain LCEL (`prompt | llm | parser`) for composability. The prompt is a zero-shot compliance evaluator — no few-shot examples needed because the requirements are explicit enough. Output JSON is validated against `ComplianceFinding` Pydantic schema post-parse, which catches hallucinated field names. Temperature=0 for determinism.

**Alternatives considered:**
- `instructor` library — structured output via OpenAI function-calling protocol; cleaner but requires an OpenAI-compatible endpoint
- Ollama's native `/api/generate` with `format: "json"` — forces JSON output at the model level; less portable across LLM backends
- GPT-4o via OpenAI API — much better compliance reasoning but costs money and needs internet; not appropriate for auditing sensitive filings

---

### ⬜ Task 8: Compliance Requirements (`app/audit/requirements.py`)

**What we'll build:** A list of 7 SEC rules we check every filing against.

#### Like I'm 5
It's like a report card with 7 questions. "Did they write about their risks? Did they explain their money? Did they talk about lawsuits?" Each question has a search phrase we use to find the right sticky notes.

#### Adult dev
A dataclass or typed dict per requirement: `{name, query, weight, regulation_ref}`. 7 SEC Reg S-K items: 1A (Risk Factors), 7 (MD&A), 7 Liquidity, ASC 606 Revenue, Item 3 Legal, Item 8 Auditor, Going Concern. Weights sum to 1.0 for the compliance score formula.

#### Master's student
Requirements map to Reg S-K disclosure items. Weight assignment reflects materiality: MD&A (Item 7) and Risk Factors (1A) carry higher weight because SEC enforcement actions disproportionately cite deficiencies there. The query string per requirement is a retrieval hint — it's embedded at query time, so it should use the same vocabulary as filing authors, not regulatory citation language (e.g., "revenue recognized when performance obligation satisfied" not "ASC 606-10-25-1").

---

### ⬜ Task 9: Grounding Checker (`app/audit/grounding.py`)

**What we'll build:** After the AI gives its answer, verify the quote it cited actually appears in the text we gave it.

#### Like I'm 5
The AI says "I found this sentence: '...' on page 5." We go back to our sticky notes and check: is that sentence actually there? If it's not, we mark the AI's answer as untrustworthy.

#### Adult dev
`thefuzz.fuzz.partial_ratio(evidence, context_string)` — fuzzy string match. If score < `grounding_min_ratio` (default 75), mark finding as ungrounded. Returns `(is_grounded: bool, best_ratio: int)`. This is the hallucination firewall.

#### Master's student
`partial_ratio` handles substring matching — it scores the best alignment of the shorter string within the longer. This is correct because LLMs paraphrase or slightly truncate quotes. Threshold 75 was chosen empirically: below 75 usually means the LLM invented the quote; above 75, it's a real excerpt with minor rephrasing. `thefuzz` uses Levenshtein distance under the hood (via `rapidfuzz` if installed for C-speed). Alternative: semantic similarity between evidence and retrieved chunks via cosine distance — more robust to paraphrasing but adds another embedding call per finding.

---

### ⬜ Task 10: Audit Loop (`app/audit/auditor.py`)

**What we'll build:** Wire everything together — retrieve → ask AI → check grounding → score.

#### Like I'm 5
This is the brain of the whole system. It loops through all 7 questions, for each one: finds the sticky notes, asks the AI, checks if the AI is telling the truth, and writes down a grade.

#### Adult dev
`audit(doc_id) -> ComplianceScorecard`: iterates requirements, calls `retrieve → assemble_context → llm_chain → grounding_check` per requirement. Builds `ComplianceFinding` list. Score = `Σ(weight × confidence × pass_binary) / Σ(weight) × 100`.

#### Master's student
The scoring formula is a weighted confidence-adjusted pass rate. `pass_binary` is 1 for Pass, 0 for Fail/Insufficient. Multiplying by `confidence` penalizes low-certainty passes (the LLM said pass but wasn't sure). This avoids the binary trap where a borderline finding scores the same as a definitive one. The auditor is synchronous — for production, you'd make retrieval and LLM calls async (`asyncio.gather`) to parallelize the 7 requirement checks.

---

### ⬜ Task 11: Report Storage + Doc Registry (`app/storage/storage.py`, `app/storage/doc_registry.py`)

**What we'll build:** Save audit results to disk as JSON, and keep a registry of which PDFs have been uploaded.

#### Like I'm 5
After we grade the report card, we save it in a folder. We also keep a list of all the company files we've seen so we don't analyze the same one twice.

#### Adult dev
`storage.py`: serialize `ComplianceScorecard` → JSON, write to `./data/reports/{report_id}.json`. `doc_registry.py`: read/write a `./data/registry.json` mapping `doc_id → {company, filing_year, doc_name, uploaded_at}`.

#### Master's student
Plain JSON file storage is intentional for this scope — no database dependency, no migration overhead, easy to inspect. In production: Postgres + SQLAlchemy for the registry (queryable, transactional); S3 or GCS for report blobs. The `doc_id` UUID is the join key between the registry, ChromaDB metadata, and report filenames — all three stores are keyed by the same UUID, so any store can be queried independently.

---

### ⬜ Task 12: FastAPI Routes (`app/api/routes.py`, `app/main.py`)

**What we'll build:** An HTTP API so anything (a script, a web app, a recruiter's curl command) can upload a PDF and get a compliance report back.

#### Like I'm 5
Instead of running a Python script yourself, you send a file over the internet and get your report card back. It's like a vending machine — put PDF in, get report out.

#### Adult dev
4 endpoints: `POST /upload` (multipart PDF → `UploadResponse`), `POST /audit` (trigger audit job → `AuditResponse`), `GET /report/{id}` (fetch scorecard JSON), `GET /health`. `python-multipart` handles file upload. Pydantic models validate request/response.

#### Master's student
FastAPI uses Starlette's ASGI foundation — `POST /audit` returns immediately with a `report_id` while the audit runs. In this MVP, the audit is synchronous (runs in the same request). Production path: queue the audit job (Celery + Redis or FastAPI BackgroundTasks), poll `GET /report/{id}` until status changes from `pending` to `complete`. This is the classic async job pattern for CPU/IO-bound work behind HTTP.

---

### ⬜ Task 13: End-to-End Smoke Test

**What we'll build:** One test that does the whole flow: upload PDF → audit → fetch report → check score.

#### Like I'm 5
We pretend to be a real user and do everything from start to finish, just to make sure nothing is broken when all the pieces are connected.

#### Adult dev
`TestClient(app)` from FastAPI/Starlette. Upload the `sample_pdf_path` fixture, fire `/audit`, poll `/report/{id}`, assert `score >= 0` and `len(findings) == 7`.

#### Master's student
Integration test vs. unit test: this one doesn't mock anything except Ollama (still no LLM dependency in CI). ChromaDB runs in-memory (`chroma_persist_dir=":memory:"`) via env override in the test. This is the confidence gate before Docker packaging — if this passes, the API surface contract is correct.

---

### ⬜ Task 14: Evaluation Script (`scripts/evaluate.py`)

**What we'll build:** A script to measure how good the system is at finding compliance issues, given a set of labeled filings.

#### Like I'm 5
Imagine you have 10 report cards already graded by a teacher. You run your robot grader on the same papers and see how often it agrees with the teacher.

#### Adult dev
Compare system findings against a ground-truth JSON fixture. Compute per-requirement precision/recall, macro-averaged F1. Print a table. Not a pytest test — a CLI script you run manually against real filings.

#### Master's student
This is an offline eval harness. The ground-truth labels are human-annotated `{doc_id, requirement, expected_status}` tuples. F1 per requirement is more informative than overall accuracy because requirements have different base rates (most filings pass Auditor Info; many fail Going Concern). ROUGE-L on evidence strings could measure extraction quality separately from classification quality. This is the artifact that goes in the resume — "evaluated on N filings, X% F1."

---

### ⬜ Task 15: README + Architecture Diagram

**What we'll build:** The front door of the repo — explains what it is, how to run it, and has a diagram of how the pieces connect.

#### Like I'm 5
This is the instruction manual. It tells anyone who opens the project: "here's what this is, here's how to run it, here's a picture of how it works."

#### Adult dev
`README.md`: badges (Python version, license), 2-sentence description, architecture diagram (ASCII or Mermaid), quickstart (`uv sync`, `ollama pull`, `uvicorn`), API examples with curl.

#### Master's student
The architecture diagram is the resume artifact. It should show the data flow: PDF → Loader → Chunker → Embedder → ChromaDB; Query → Retriever → Context Assembler → LLM → Grounding Checker → Scorecard. Mermaid renders inline on GitHub. The README is the first thing a hiring manager reads — it should answer "what is this, does it work, can I verify it" in under 60 seconds.

---

## KEY DECISIONS LOG

| Decision | What we chose | Why | Rejected alternative |
|---|---|---|---|
| LLM | mistral:7b-instruct via Ollama | Free, local, no API key, offline | GPT-4o (costs money, needs internet) |
| Embeddings | all-MiniLM-L6-v2 | Fast, local, MIT license, 384-dim fine for retrieval | OpenAI embeddings (costs money) |
| Vector DB | ChromaDB | In-process, no server, persists to disk | FAISS (no metadata filter), Qdrant (needs server) |
| PDF parsing | pypdf | Pure Python, MIT license, good on EDGAR filings | pymupdf (GPL), unstructured (heavy) |
| Chunking | RecursiveCharacterTextSplitter | Respects paragraph/line breaks, LangChain built-in | Semantic chunking (2× slower), token-based (ignores structure) |
| Hallucination guard | thefuzz partial_ratio | Simple, no extra model call, 0-100 score | Cosine similarity (needs embedding call) |
| Package manager | uv | Fast, reproducible uv.lock, modern pyproject.toml | pip + requirements.txt (no lock), Poetry (slower) |
| API framework | FastAPI | Pydantic-native, async-ready, auto OpenAPI docs | Flask (no async, no built-in validation) |
| Testing | pytest + mocks for external deps | Fast CI, no Ollama/ChromaDB needed in test | Integration-only tests (slow, fragile) |
