# RAG Financial Compliance Auditor

A retrieval-augmented system that checks SEC filings (10-K PDFs) against a list of compliance requirements and produces source-grounded findings, meaning every answer points back to the exact page and quote it came from.

I built this out of curiosity, wanting to try out RAG, chunking and retrieval myself with no constraints, just to understand how the pieces actually fit together.

## What this is (non-technical)

Compliance reviewers spend a lot of time reading long financial filings just to check whether a handful of specific requirements are met somewhere in the document. This project automates that first pass.

You give it a filing and a list of requirements ("does the filing disclose X", "is Y mentioned in the risk section"). It reads the document, finds the relevant passages, and returns a verdict for each requirement: Pass, Fail, or Insufficient Evidence, along with the exact quote and page number it used to decide. Nothing is answered from memory. If the model can't find real evidence in the document, it says so instead of guessing.

The point isn't to replace a human reviewer. It's to do the tedious part, scanning hundreds of pages for specific facts, so a reviewer can spend their time judging the results instead of hunting for them.

Current state: the full pipeline works end to end, from uploading a filing to getting back a scorecard with page-cited findings. An evaluation framework to measure retrieval and grounding quality more rigorously is still on the list.

## What this is (technical)

Standard RAG architecture, built to keep every generated claim traceable back to a source.

**Ingestion**
- `app/ingestion/loader.py`: extracts text per page from a PDF with `pypdf`, keeps page numbers as metadata
- `app/ingestion/chunker.py`: splits pages into overlapping chunks (`RecursiveCharacterTextSplitter`), tags each chunk with a stable `chunk_id`
- `app/ingestion/embedder.py`: embeds chunks with a sentence-transformer model (`all-MiniLM-L6-v2`) and stores them in a local ChromaDB collection

**Retrieval**
- `app/retrieval/retriever.py`: similarity search against the vector store, scoped per document
- `app/retrieval/context.py`: reassembles retrieved chunks into a page-tagged context block for the prompt

**Generation**
- `app/llm/client.py`: local LLM via Ollama (`mistral:7b-instruct`), temperature 0, forced JSON output
- `app/llm/prompts.py`: the compliance prompt requires every finding to include a verbatim quote from the given context and a page number, and to mark a requirement as "Insufficient Evidence" rather than invent an answer
- `app/models.py`: Pydantic schemas for a single finding and the aggregate scorecard, so output is validated, not just parsed

**Audit orchestration**
- `app/audit/auditor.py`: runs each requirement through retrieval and generation, then double-checks the model's own claim: it fuzzy-matches the returned evidence quote against the actual retrieved context, and downgrades the finding to "Insufficient Evidence" if the quote doesn't really appear there. The model is not trusted blindly, even though the prompt already asks it to behave.
- `app/audit/requirements.py`: a default set of generic 10-K disclosure checks, so the system runs out of the box without needing a custom requirement list first

**API**
- `app/api/main.py`: three endpoints. `/upload` takes a PDF, chunks and embeds it, returns a `doc_id`. `/audit` takes a `doc_id` (and optional custom requirements), runs the audit synchronously, and returns the scorecard directly. `/health` for a basic liveness check. No job queue or async status polling, since a single filing audits fast enough to just wait for it.

**Why local model instead of a cloud API:** filings can contain sensitive or embargoed financial data. Keeping the LLM local during development avoids sending that data to a third party.

**Why grounding matters here:** in compliance, a hallucinated citation is worse than no answer. The prompt asks the model to quote the source or admit it found nothing, and the auditor verifies that quote against the real context instead of just trusting it.

**Tested:** loader, chunker, embedder, retriever, prompt formatting, the auditor's grounding check, and the API endpoints all have unit tests (`tests/`).

**Not yet built:** a proper evaluation framework to measure retrieval and grounding quality against a labeled test set, rather than spot-checking by hand.

## Stack

Python, LangChain, ChromaDB, sentence-transformers, Ollama, FastAPI, Pydantic, pytest.
