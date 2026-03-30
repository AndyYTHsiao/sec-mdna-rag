import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, TextIO
from utils import hash_config


def _count_tokens(text: str) -> int:
    """
    Count the number of tokens by splitting the text.

    Args:
        text (str): The text to be split.

    Returns:
        The number of tokens in the text.
    """
    return len(text.split())


def _write_chunk(
    current_chunk: list[dict],
    chunk_index: int,
    fout: TextIO,
    first_write: bool,
) -> bool:
    """
    Save the current chunk.

    Args:
        current_chunk (list[dict]): The currently processed chunk.
        chunk_index (int): The index of current chunk.
        fout (TextIO): An open writable text stream where the chunk will be written.
        first_write (bool): Whether the chunk is the first written chunk.

    Returns:
        Flag indicating the first chunk has been written.
    """
    first = current_chunk[0]

    chunk = {
        "chunk_id": f"{first['doc_id']}_{chunk_index:02d}",
        "section": first["section"],
        "chunk_index": chunk_index,
        "source_paragraphs": [p["paragraph_id"] for p in current_chunk],
        "text": " ".join(p["text"] for p in current_chunk),
    }

    if not first_write:
        fout.write("\n")
    fout.write(json.dumps(chunk, ensure_ascii=False))

    return False


def _iterate_rows(filings_root: Path) -> Iterable[dict]:
    """
    Iterate over MD&A paragraph rows from filing files.

    This generator traverses a directory structure organized by CIK,
    reads ``.jsonl`` filing files, sorts rows within each file by
    ``paragraph_id``, filters for MD&A sections, and yields rows
    one at a time.

    Directory structure is expected to be:

        filings_root/
            <CIK>/
                <filing>.jsonl
                ...

    Each ``.jsonl`` file should contain one JSON object per line.

    Args:
        filings_root (Path):
            Root directory containing subdirectories named by CIK.
            Each CIK directory should contain ``.jsonl`` filing files.

    Yields:
        dict:
            A dictionary representing a single MD&A paragraph row.
            Each yielded row is expected to contain at least:

            - ``paragraph_id`` (int):
                Paragraph ordering identifier used for sorting.
            - ``section`` (str):
                Section label used to filter MD&A content.
            - Other keys depend on the source dataset.

    Notes:
        - Rows are sorted within each file by ``paragraph_id`` before
          yielding.
        - Only rows whose ``section`` contains the substring
          ``"MD&A"`` are yielded.
        - Empty lines in input files are ignored.
        - Non-directory entries under ``filings_root`` are skipped.
    """
    for cik_dir in tqdm(sorted(filings_root.iterdir()), desc="CIKs", leave=False):
        if not cik_dir.is_dir():
            continue

        cik = cik_dir.name

        for filing_path in sorted(cik_dir.glob("*.jsonl")):
            # Read and sort rows by paragraph_id
            rows = []
            with open(filing_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows.append(row)

            rows.sort(key=lambda r: r["paragraph_id"])

            for row in tqdm(
                rows, desc=f"{cik}/{filing_path.name}", unit="para", leave=False
            ):
                # Filter only MD&A paragraphs
                if "MD&A" not in row.get("section", ""):
                    continue

                yield row


def document_chunking_stream(
    filings_dir: Path,
    output_path: str,
    target_section: str = "MD&A",
    max_tokens: int = 300,
    max_paragraphs: int = 2,
) -> None:
    """
    Stream and chunk filing paragraphs into structured document chunks.

    This function reads paragraph-level filing data from ``filings_dir``,
    groups paragraphs by document and section, and writes chunked output
    incrementally to ``output_path``. Chunk boundaries are determined by
    token count and paragraph count thresholds.

    Paragraphs are processed sequentially and written immediately when
    chunk limits are reached, enabling memory-efficient processing of
    large datasets.

    Args:
        filings_dir (Path):
            Root directory containing filing data organized by CIK.
            The directory structure is expected to match the input
            requirements of ``_iterate_rows``.

        output_path (str):
            File path where chunked output will be written.
            The file will be opened in write mode (``"w"``) and
            overwritten if it already exists.

        target_section (str, default="MD&A"):
            Section name to include during chunking. All rows are
            expected to belong to this section.

        max_tokens (int, default=300):
            Maximum number of tokens allowed in a single chunk.
            When the token count reaches or exceeds this limit,
            the current chunk is written and reset.

        max_paragraphs (int, default=2):
            Maximum number of paragraphs allowed in a single chunk.
            When this limit is reached, the chunk is written even
            if the token limit has not been exceeded.

    Raises:
        AssertionError:
            If any row violates expected schema constraints, including:

            - ``section`` does not match ``target_section``
            - ``text`` is empty
            - ``paragraph_id`` is missing

    Notes:
        - Chunking resets when a new document (``doc_id``) or section
          is encountered.
        - Both token count and paragraph count limits are enforced.
          The first limit reached triggers chunk writing.
        - Output is written incrementally using ``_write_chunk``,
          minimizing memory usage.
        - Token counting is performed using ``_count_tokens``.

    Expected Row Schema:
        Each row yielded from ``_iterate_rows`` must contain:

        - ``doc_id`` (str):
            Unique identifier for the document.
        - ``section`` (str):
            Section label (must match ``target_section``).
        - ``paragraph_id`` (int):
            Paragraph ordering identifier.
        - ``text`` (str):
            Paragraph content used for token counting.

    Side Effects:
        Writes chunked data to ``output_path``.
    """
    current_doc = None
    current_section = None
    current_chunk: list[dict] = []
    token_count = 0
    chunk_index = 0
    first_write = True

    rows = _iterate_rows(filings_dir)
    with open(output_path, "w", encoding="utf-8") as fout:
        for row in rows:
            assert row["section"] == target_section, (
                f"Unfound target section in row: {target_section}."
            )
            assert row["text"].strip(), "Empty text in the row."
            assert "paragraph_id" in row, "Missing paragraph ID."

            if current_doc != row["doc_id"] or current_section != row["section"]:
                if current_chunk:
                    chunk_index += 1
                    first_write = _write_chunk(
                        current_chunk, chunk_index, fout, first_write
                    )

                current_chunk = []
                token_count = 0
                chunk_index = 0
                current_doc = row["doc_id"]
                current_section = row["section"]

            current_chunk.append(row)
            token_count += _count_tokens(row["text"])

            if token_count >= max_tokens or len(current_chunk) >= max_paragraphs:
                chunk_index += 1
                first_write = _write_chunk(
                    current_chunk, chunk_index, fout, first_write
                )
                current_chunk = []
                token_count = 0

        # Final flush
        if current_chunk:
            chunk_index += 1
            _write_chunk(current_chunk, chunk_index, fout, first_write)


if __name__ == "__main__":
    data_dir = Path("./data")
    parser = argparse.ArgumentParser(description="Chunk MD&A sections from filings")
    parser.add_argument(
        "--filings-dir",
        type=Path,
        default=data_dir / Path("filings"),
        help="Directory of raw filings",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=700, help="Maximum tokens per chunk"
    )
    parser.add_argument(
        "--max-paragraphs", type=int, default=1, help="Maximum paragraphs per chunk"
    )
    args = parser.parse_args()

    filings_dir = args.filings_dir
    max_tokens = args.max_tokens
    max_paragraphs = args.max_paragraphs

    corpus_cfg = {
        "filings_dir": str(filings_dir),
        "max_tokens": max_tokens,
        "max_paragraphs": max_paragraphs,
    }
    corpus_hash = hash_config(corpus_cfg)
    output_path = data_dir / Path(
        f"corpus_t{max_tokens}_p{max_paragraphs}_{corpus_hash}.jsonl"
    )

    document_chunking_stream(
        filings_dir=filings_dir,
        output_path=output_path,
        max_tokens=max_tokens,
        max_paragraphs=max_paragraphs,
    )
