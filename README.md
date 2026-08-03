# RAG System for SEC 10-K MD&A Analysis

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions about the Management's Discussion and Analysis (MD&A) sections of SEC Form 10-K filings across multiple companies.
The system provides grounded financial question answering with transparent citations to source paragraphs.


### Key features

* Hybrid retrieval using FAISS, BM25, and Reciprocal Rank Fusion
* Optional metadata-based chunk filtering
* Paragraph-level source citations
* Interactive CLI and Streamlit interfaces
* Transparent retrieval scores and ranked results
* A real-world corpus of SEC Form 10-K filings

### System Architecture

The retrieval pipeline combines dense and sparse search:

1. Dense retrieval using FAISS vector search
2. Sparse retrieval using BM25 keyword search
3. Chunk filtering to improve retrieval quality
4. Reciprocal Rank Fusion (RRF) combines the dense and sparse rankings
5. Top-ranked chunks are passed to the LLM
6. The LLM generates a grounded answer with paragraph-level citations

### Workflow
```
User Query
     ↓
Query Processing
     ↓
Score All Chunks
     ↓
Chunk filtering (Optional)
     ↓
Dense and Sparse Retrieval
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
Python 3.14 or later is required.

### Prerequisites
1. Clone the repo and switch to its directory
     ```
     git clone https://github.com/AndyYTHsiao/sec-mdna-rag.git
     cd sec-mdna-rag
     ```
2. Install the dependencies using one of the following commands:
     ```
     # Install dependencies and update the lock file if necessary
     uv sync

     # Alternatively, install exactly from the existing lock file
     uv sync --frozen
     ```
3. Create a `.env` file with your OpenAI API key saved in the following format:
     ```
     OPENAI_API_KEY=YOUR_API_KEY
     ```

### Run the CLI
```
uv run python -m src.cli
```

### Run the Streamlit app
```
uv run python -m streamlit run src/app.py
```

## Example Query and Output

**Question:**
What was the primary driver of the increase in Mac net sales in 2020?

**Answer:**

Higher net sales of the MacBook Pro. (Document 1)

Retrieved Chunks:

Chunk 1 🥇 Top Match | ID: 320193_2020-09-26_11

Content:
Mac net sales increased during 2020 compared to 2019 due primarily to higher net sales of MacBook Pro.

## Dataset Description

All raw data is stored in ./data/filings, with each folder named after the company's Central Index Key (CIK).
The files use the JSON Lines (.jsonl) format, with each line representing one paragraph from Item 7, "Management’s Discussion and Analysis of Financial Condition and Results of Operations."
The dataset contains:
- 20 companies
- SEC Form 10-K filings covering fiscal years 2020–2023
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

## Evaluation

[`eval_dataset`](./data/eval_dataset.json) contains 100 evaluation queries stored as an array of JSON objects.
Each entry in the dataset has the following format:

```json
{
     "id": "The query ID (formatted as q_xxx)",
     "query": "The query",
     "company": "The company that the query refers to",
     "evidence": [
          {
               "doc_id": "Document ID",
               "paragraph_id": "Paragraph ID",
               "text": "The supporting text from the corpus",
               "supports": "The degree to which the evidence supports the query (full or partial)"
          }
     ],
     "type": "The query type (factoid, reasoning, or multi-hop)"
}
```

To run the ablation study, use the following command:
```
uv run python -m src.run_ablation
```

The evaluation results are summarized in the following table.
**Unfiltered** indicates that chunk filtering is disabled, whereas **Filtered** indicates that it is enabled.
**Boldface** indicates the best result in each metric column.

In the ablation study, each retrieval chunk corresponds to one paragraph in the original filing.
Retrieval is evaluated using Recall@K and NDCG@K over 100 manually curated queries.
Evidence labeled full or partial is assigned a relevance weight when calculating NDCG.
Reported scores are averaged across all queries.
Relative gain is calculated as
$
\frac{(\text{Filtered} − \text{Unfiltered})}{\text{Unfiltered}} × 100\%.
$

| Retrieval Method | Setting           |              Recall@1 |              Recall@5 |             Recall@10 |                NDCG@1 |                NDCG@5 |               NDCG@10 |
| :--------------- | :---------------- | --------------------: | --------------------: | --------------------: | --------------------: | --------------------: | --------------------: |
| **Sparse**       | Unfiltered        |                0.3592 |                0.5350 |                0.5917 |                0.4000 |                0.4652 |                0.4850 |
|                  | Filtered          |                0.4833 |                0.7345 |                0.7857 |                0.5600 |                0.6546 |                0.6736 |
|                  | **Gain (Δ / %Δ)** | **+0.1241 / +34.55%** | **+0.1995 / +37.29%** | **+0.1940 / +32.79%** | **+0.1600 / +40.00%** | **+0.1894 / +40.71%** | **+0.1886 / +38.89%** |
| **Dense**        | Unfiltered        |                0.3178 |                0.6078 |                0.7075 |                0.3800 |                0.5065 |                0.5441 |
|                  | Filtered          |                0.4077 |                0.7710 |            **0.8670** |                0.5000 |                0.6431 |                0.6815 |
|                  | **Gain (Δ / %Δ)** | **+0.0899 / +28.29%** | **+0.1632 / +26.85%** | **+0.1595 / +22.54%** | **+0.1200 / +31.58%** | **+0.1366 / +26.97%** | **+0.1374 / +25.25%** |
| **Hybrid**       | Unfiltered        |                0.3787 |                0.6803 |                0.7262 |                0.4600 |                0.5708 |                0.5889 |
|                  | Filtered          |            **0.5365** |            **0.7918** |                0.8395 |            **0.6400** |            **0.7200** |            **0.7407** |
|                  | **Gain (Δ / %Δ)** | **+0.1578 / +41.67%** | **+0.1115 / +16.39%** | **+0.1133 / +15.60%** | **+0.1800 / +39.13%** | **+0.1492 / +26.14%** | **+0.1518 / +25.78%** |


Overall, chunk filtering improves every reported retrieval metric across sparse, dense, and hybrid retrieval.
Hybrid retrieval with filtering achieves the best Recall@1, Recall@5, and NDCG scores, while filtered dense retrieval achieves the highest Recall@10.
The largest relative improvement is observed for hybrid Recall@1, which increases by 41.67%.

## Future Improvements

- Add cross-encoder reranking for improved precision