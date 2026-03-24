#!/usr/bin/env python3
"""
Step 1.2 — Document Chunker

Reads data/raw_sources.jsonl (from Step 1.1) and splits each document into
~800-token overlapping chunks with 200-token overlap using tiktoken.

Output: data/chunks.jsonl
Each line: {chunk_id, source_url, project, doc_type, title, chunk_text}

Usage:
    python scripts/chunk_sources.py [--input data/raw_sources.jsonl] [--output data/chunks.jsonl]
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import tiktoken

CHUNK_SIZE = 800
OVERLAP = 200
ENCODING_NAME = "cl100k_base"
MIN_CHUNK_TOKENS = 40


def tokenize(text: str, enc: tiktoken.Encoding) -> list[int]:
    return enc.encode(text)


def chunk_text(
    text: str,
    enc: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """Split text into overlapping token-bounded chunks, breaking on paragraph/sentence boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current_tokens: list[int] = []
    current_paras: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = enc.encode(para)

        if len(current_tokens) + len(para_tokens) > chunk_size and current_tokens:
            chunk_text_str = "\n\n".join(current_paras)
            chunks.append(chunk_text_str)

            overlap_tokens = current_tokens[-overlap:] if len(current_tokens) > overlap else current_tokens
            overlap_text = enc.decode(overlap_tokens)

            current_paras = [overlap_text]
            current_tokens = list(overlap_tokens)

        current_paras.append(para)
        current_tokens.extend(para_tokens)

    if current_tokens and len(current_tokens) >= MIN_CHUNK_TOKENS:
        chunks.append("\n\n".join(current_paras))

    if not chunks and text.strip():
        tokens = enc.encode(text.strip())
        if len(tokens) >= MIN_CHUNK_TOKENS:
            chunks.append(text.strip())

    return chunks


def make_chunk_id(source_url: str, idx: int) -> str:
    h = hashlib.md5(source_url.encode()).hexdigest()[:8]
    return f"{h}_{idx:04d}"


def main():
    parser = argparse.ArgumentParser(description="Chunk raw Kubeflow sources into ~800-token windows")
    parser.add_argument("--input", default="data/raw_sources.jsonl")
    parser.add_argument("--output", default="data/chunks.jsonl")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"ERROR: Input file not found: {args.input}")

    enc = tiktoken.get_encoding(ENCODING_NAME)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            total_docs += 1

            chunks = chunk_text(doc["content"], enc, args.chunk_size, args.overlap)
            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id": make_chunk_id(doc["url"], i),
                    "source_url": doc["url"],
                    "project": doc["project"],
                    "doc_type": doc["doc_type"],
                    "title": doc["title"],
                    "chunk_text": chunk,
                }
                fout.write(json.dumps(record) + "\n")
                total_chunks += 1

    print(f"Processed {total_docs} documents → {total_chunks} chunks")
    print(f"Avg chunks/doc: {total_chunks / max(total_docs, 1):.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
