#!/usr/bin/env python3
"""Build BM25 index for merged corpus without Qdrant."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv

load_dotenv()
os.environ["USE_MERGED_CORPUS"] = "1"

from config import Config
from main.bm25_retriever import BM25Retriever
from utils.data_loader import LegalDataLoader


def main() -> None:
    t0 = time.time()
    print(f"Corpus: {Config.CORPUS_PATH}")
    print(f"BM25 index: {Config.BM25_INDEX_FILE}")

    loader = LegalDataLoader()
    loader.load_legal_corpus()
    documents = loader.prepare_documents_for_indexing()
    print(f"Articles: {len(documents):,}")

    bm25 = BM25Retriever()
    bm25.build_index(documents)
    bm25.save_index()
    stats = bm25.get_index_stats()
    print(f"Done in {time.time() - t0:.1f}s")
    print(stats)


if __name__ == "__main__":
    main()
