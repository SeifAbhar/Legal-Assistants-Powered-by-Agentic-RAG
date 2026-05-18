"""Utility to chunk documents into token-sized pieces (default 800 tokens).

This module uses tiktoken when available for accurate tokenization (recommended).
It falls back to a word-based approximation if tiktoken is not installed.

Functions:
- chunk_text(text, chunk_size=800, overlap=0, model='gpt-4o-mini'): returns list of chunks (strings).
- chunk_documents(docs, ...): docs is iterable of (id, text) or dicts with 'id'/'text'. Returns list of chunk dicts with metadata.

CLI:
    python -m utils.token_chunker --input-dir data/txt --output-file chunks.jsonl --chunk-size 800

Output format (JSONL): {"doc_id": "<id>", "chunk_index": 0, "text": "...", "tokens": 800}

"""
import argparse
import json
import os
from typing import Iterable, List, Dict, Tuple, Union


def _get_tiktoken_encoding(model: str):
    try:
        import tiktoken
    except Exception:
        return None

    try:
        # Prefer model-specific encoding when possible
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback to the commonly-used cl100k_base encoding
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 0, model: str = "gpt-4o-mini") -> List[str]:
    """Chunk a single text string into token-sized pieces.

    Args:
        text: the input document text.
        chunk_size: maximum tokens per chunk.
        overlap: number of tokens to overlap between consecutive chunks.
        model: model name to select tokenizer (used by tiktoken).

    Returns:
        list of chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    encoding = _get_tiktoken_encoding(model)
    if encoding is not None:
        # Accurate tokenization using tiktoken
        try:
            tokens = encoding.encode(text)
        except Exception:
            # If encode fails for some reason, fall back to naive split
            encoding = None
        else:
            chunks: List[str] = []
            step = chunk_size - overlap
            if step <= 0:
                step = chunk_size
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i : i + chunk_size]
                # decode back to text
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
            return chunks

    # Fallback: approximate with word-based splitting
    words = text.split()
    # Treat each 'word' as an approximate token
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
    return chunks


def chunk_documents(
    docs: Iterable[Union[Tuple[str, str], Dict[str, str]]],
    chunk_size: int = 800,
    overlap: int = 0,
    model: str = "gpt-4o-mini",
) -> List[Dict]:
    """Chunk multiple documents.

    docs can be an iterable of (doc_id, text) tuples OR an iterable of dicts with keys 'id' and 'text'.

    Returns a list of dicts: {"doc_id": ..., "chunk_index": i, "text": ..., "tokens": n}
    """
    out = []
    for raw in docs:
        if isinstance(raw, dict):
            doc_id = raw.get("id") or raw.get("doc_id")
            text = raw.get("text")
        else:
            doc_id, text = raw
        if text is None:
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, model=model)
        for idx, c in enumerate(chunks):
            # Determine approximate token count: use tokenizer if available
            encoding = _get_tiktoken_encoding(model)
            if encoding is not None:
                try:
                    n_tokens = len(encoding.encode(c))
                except Exception:
                    n_tokens = len(c.split())
            else:
                n_tokens = len(c.split())
            out.append({"doc_id": doc_id, "chunk_index": idx, "text": c, "tokens": n_tokens})
    return out


def _gather_txt_files(input_dir: str) -> Iterable[Tuple[str, str]]:
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(".txt"):
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc_id = os.path.relpath(path, input_dir)
                yield (doc_id, text)


def main():
    parser = argparse.ArgumentParser(description="Chunk documents into token-sized chunks (default 800).")
    parser.add_argument("--input-dir", type=str, help="Directory containing .txt files to chunk.")
    parser.add_argument("--output-file", type=str, help="Path to output JSONL file (one chunk per line).")
    parser.add_argument("--chunk-size", type=int, default=800, help="Max tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping tokens between chunks.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name for tiktoken encoding selection.")
    args = parser.parse_args()

    if not args.input_dir or not args.output_file:
        parser.print_help()
        return

    docs = list(_gather_txt_files(args.input_dir))
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, overlap=args.overlap, model=args.model)

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {args.output_file}")


if __name__ == "__main__":
    main()
