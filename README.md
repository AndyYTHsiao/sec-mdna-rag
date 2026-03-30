# RAG System for SEC 10-K MD&A Analysis

## Overview

This project builds a Retrieval-Augmented Generation (RAG) system for querying Management Discussion & Analysis (MD&A) sections from SEC Form 10-K filings across multiple companies.
The system enables grounded financial question answering with transparent citation of source paragraphs.

### Key features

* Hybrid retrieval (FAISS + BM25 + Reciprocal Rank Fusion)
* Chunk-level citation grounding
* Streamlit and command line interactive interface
* Retrieval transparency with document scoring
* Real-world financial corpus (SEC 10-K filings)

### Workflow
```
User Query
     ↓
Query Processing
     ↓
Hybrid Retrieval (FAISS + BM25)
     ↓
Reciprocal Rank Fusion
     ↓
Top-K Selection
     ↓
Context Assembly
     ↓
LLM Answer Generation
     ↓
Answer + Citations
```

## Quick Start
All dependencies are managed by [uv](https://docs.astral.sh/uv/).
Python 3.14+ is required for the system.

### Prerequisites
1. Clone the repo and switch to its directory
     ```
     git clone https://github.com/AndyYTHsiao/sec-mdna-rag.git
     cd sec-mdna-rag
     ```
2. Intialize the project
     ```
     uv sync             # If you want the lock to be updated
     uv sync --frozen    # If you want to use the lock file as truth
     ```
3. Create a `.env` file with your OpenAI API key saved in the following format:
     ```
     OPENAI_API_KEY=YOUR_API_KEY
     ```

### Run the CLI
```
uv run src/cli.py
```

### Run the Streamlit app
```
uv run streamlit run src/app.py
```

## Example Query and Output

**Question:**
What was the primary driver of the increase in Mac net sales in 2020?

**Answer:**
Higher net sales of the MacBook Pro. (Document 1: 320193_2020-12-31_11)

**Retrieved Documents:**
Document 1 (Rank 1)  
Mac net sales increased during 2020 compared to 2019 due primarily to higher net sales of MacBook Pro.

## Dataset Description
All raw data is saved in `./data/filings`, with folder named after the Central Index Key (CIK) of each company.
All files are in `jsonl` format, and each line represents a paragraph from the MD&A section (Item 7).
Together, the raw data contains:
- 20 companies
- 4 years of 10-K filings (2020–2023)
- ~8,946 MD&A paragraphs

### Example structure
```json
{
    "doc_id": "...",
    "cik": "...",
    "ticker": "...",
    "company": "...",
    "sector": "...",
    "industry": "...",
    "fiscal_year": "...",
    "fiscal_year_end": "...",
    "section": "...",
    "paragraph_id": "...",
    "text": "..."
}
```
## System Architecture

The retrieval pipeline combines dense and sparse search:

1. Dense retrieval using FAISS vector search
2. Sparse retrieval using BM25 keyword search
3. Reciprocal Rank Fusion (RRF) combines rankings
4. Top-ranked chunks are passed to the LLM
5. The LLM generates grounded responses with citations

## Future Improvements

- Add cross-encoder reranking for improved precision
- Implement automated retrieval evaluation metrics
- Improve temporal filtering (year-aware retrieval)