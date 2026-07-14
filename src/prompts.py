# =================
# Query database
# =================


QUERY_DB_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions using the provided context.


"Follow these rules strictly:

1. Carefully read the context before answering.
2. If the answer exists in the context:
    - Use the information from the context.
    - Cite the document number.
3. If the answer does NOT exist in the context:
    - Answer from general knowledge.
    - Start your answer with:
    'This answer is based on my general knowledge.
4. Never fabricate citations.
5. Prefer using context whenever possible.
"""

QUERY_DB_USER_PROMPT = """
Use the following context to answer the question.


Context:
{context}


Question:
{query}
"""

# =================
# Search metadata
# =================

FILTER_CHUNK_SYSTEM_PROMPT = """
You extract retrieval filters from financial QA queries.

Extract only:
1. company mentions that appear in the query
2. date, year, fiscal year, quarter, or date range mentioned in the query

Return valid JSON only.

Rules:
- Extract company names, abbreviations, or tickers exactly as written in the query.
- Do not infer or rewrite company names.
- Do not return CIK numbers unless they explicitly appear in the query.
- If one company is mentioned, return it as a one-item list.
- If multiple companies are mentioned, return all of them as a list.
- If no company is mentioned, return an empty list.
- Extract years such as "2020", "in 2022", "fiscal year 2019", and ranges such as "between 2020 and 2022".
- If only one year is mentioned, use it as both start_year and end_year.
- If no date is mentioned, return null for start_year, end_year, and date_text.
- Do not explain your answer.
"""

FILTER_CHUNK_USER_PROMPT = """
Extract company and date filters from the query.

Use this JSON schema:

{{
  "companies": list[str],
  "start_year": int | null,
  "end_year": int | null
}}

Examples:

Query: What were Apple's total net sales in 2023?
Output:
{{
  "companies": ["Apple"],
  "start_year": 2023,
  "end_year": 2023
}}

Query: How did AAPL describe risks related to supply chain disruptions in its 2021 annual report?
Output:
{{
  "companies": ["AAPL"],
  "start_year": 2021,
  "end_year": 2021
}}

Query: In 2022, approximately how many employees does ADM have in Ukraine?
Output:
{{
  "companies": ["ADM"],
  "start_year": 2022,
  "end_year": 2022
}}

Query: What did Amazon say about AWS operating income in Q4 2022?
Output:
{{
  "companies": ["Amazon"],
  "start_year": 2022,
  "end_year": 2022
}}

Query: Compare Costco's revenue growth between 2021 and 2023.
Output:
{{
  "companies": ["Costco"],
  "start_year": 2021,
  "end_year": 2023
}}

Query: How did AMD and Nvidia describe demand for data center products in 2023?
Output:
{{
  "companies": ["AMD", "Nvidia"],
  "start_year": 2023,
  "end_year": 2023
}}

Query: In 2020, what were the main drivers of revenue growth?
Output:
{{
  "companies": [],
  "start_year": 2020,
  "end_year": 2020
}}

Query: What did the company say about liquidity and capital resources?
Output:
{{
  "companies": [],
  "start_year": null,
  "end_year": null
}}

Now extract from this query:

Query: {query}

Output:
"""

# =================
# Main entry point
# =================


PROMPT_HELPER = {
    "query_db": {
        "system": QUERY_DB_SYSTEM_PROMPT,
        "user": QUERY_DB_USER_PROMPT,
    },
    "filter_chunk": {
        "system": FILTER_CHUNK_SYSTEM_PROMPT,
        "user": FILTER_CHUNK_USER_PROMPT,
    },
}
