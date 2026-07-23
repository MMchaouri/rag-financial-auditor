# RAG Financial Compliance Auditor

A retrieval-augmented system that checks SEC filings (10-K PDFs) against a list of compliance requirements and produces source-grounded findings, meaning every answer points back to the exact page and quote it came from.

I built this out of curiosity, wanting to try out RAG, chunking and retrieval myself with no constraints, just to understand how the pieces actually fit together.

## What this is (non-technical)

Compliance reviewers spend a lot of time reading long financial filings just to check whether a handful of specific requirements are met somewhere in the document. This project automates that first pass.

You give it a filing and a list of requirements ("does the filing disclose X", "is Y mentioned in the risk section"). It reads the document, finds the relevant passages, and returns a verdict for each requirement: Pass, Fail, or Insufficient Evidence, along with the exact quote and page number it used to decide. Nothing is answered from memory. If the model can't find real evidence in the document, it says so instead of guessing.

The point isn't to replace a human reviewer. It's to do the tedious part, scanning hundreds of pages for specific facts, so a reviewer can spend their time judging the results instead of hunting for them.

Current state: the pipeline that reads, indexes and retrieves from filings is built and tested. The layer that turns retrieved evidence into a full audit report and scorecard is still in progress.

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

**Why local model instead of a cloud API:** filings can contain sensitive or embargoed financial data. Keeping the LLM local during development avoids sending that data to a third party.

**Why grounding matters here:** in compliance, a hallucinated citation is worse than no answer. The prompt and schema both force the model to quote the source or admit it found nothing.

**Tested:** loader, chunker, embedder, retriever and prompt formatting all have unit tests (`tests/`).

**Not yet built:** the audit orchestration layer that runs a full requirement list against a filing and produces a scorecard (`app/audit/`), the API layer to trigger a run (`app/api/`), and an evaluation framework to measure retrieval and grounding quality.

## Stack

Python, LangChain, ChromaDB, sentence-transformers, Ollama, FastAPI, Pydantic, pytest.
