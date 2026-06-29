#!/usr/bin/env python3
"""
RAG Indexing Script — Full setup (Vector Store + BM25) without loading Qwen3-4B.

Usage:
    python run_indexing.py           # Normal (skip if index already exists)
    python run_indexing.py --rebuild # Force rebuild both indices
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from config import Config
from utils.data_loader import LegalDataLoader
from main.vector_store import QdrantVectorStore
from main.bm25_retriever import BM25Retriever

FORCE_REBUILD = "--rebuild" in sys.argv or "-r" in sys.argv


def log(msg):
    print(msg, flush=True)


def step(title):
    print(f"\n{'='*55}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*55}", flush=True)


# ──────────────────────────────────────────────────────────
# STEP 1: Load & prepare corpus
# ──────────────────────────────────────────────────────────
step("STEP 1 — Load legal corpus")
t0 = time.time()

loader = LegalDataLoader()
log(f"Loading corpus from: {Config.CORPUS_PATH}")
legal_docs = loader.load_legal_corpus()

if not legal_docs:
    log("❌ No legal documents loaded. Abort.")
    sys.exit(1)

log(f"✅ Loaded {len(legal_docs)} laws")

log("\nPreparing articles for indexing...")
documents = loader.prepare_documents_for_indexing()

if not documents:
    log("❌ No articles prepared. Abort.")
    sys.exit(1)

log(f"✅ Prepared {len(documents):,} articles  ({time.time()-t0:.1f}s)")


# ──────────────────────────────────────────────────────────
# STEP 2: Qdrant Vector Store
# ──────────────────────────────────────────────────────────
step("STEP 2 — Vector Store (Qdrant + Embedding)")
t1 = time.time()

log(f"Embedding model : {Config.EMBEDDING_MODEL}")
log(f"Collection name : {Config.COLLECTION_NAME}")

vector_store = QdrantVectorStore()

# Check existing collection
collections = vector_store.client.get_collections().collections
collection_exists = any(c.name == vector_store.collection_name for c in collections)

if collection_exists and not FORCE_REBUILD:
    info = vector_store.get_collection_info()
    points = info.get("points_count", 0)
    if points > 0:
        log(f"✅ Collection already has {points:,} vectors — skipping embed")
        log("   (use --rebuild to force re-embed)")
    else:
        log("Collection exists but is empty — embedding now...")
        vector_store.add_documents(documents)
        log(f"✅ Vector store built  ({time.time()-t1:.1f}s)")
else:
    if FORCE_REBUILD and collection_exists:
        log("Force rebuild — recreating collection...")
    else:
        log("Collection not found — creating...")

    vector_store.create_collection(vector_size=768, force_recreate=True)
    log(f"Embedding {len(documents):,} articles into Qdrant...")
    vector_store.add_documents(documents)
    log(f"✅ Vector store built  ({time.time()-t1:.1f}s)")


# ──────────────────────────────────────────────────────────
# STEP 3: BM25 Index
# ──────────────────────────────────────────────────────────
step("STEP 3 — BM25 Index")
t2 = time.time()

bm25 = BM25Retriever()

if not FORCE_REBUILD and bm25.load_index():
    stats = bm25.get_index_stats()
    log(f"✅ BM25 index loaded from file")
    log(f"   Documents : {stats.get('total_documents', 0):,}")
    log(f"   Vocab     : {stats.get('vocabulary_size', 0):,}")
else:
    if FORCE_REBUILD:
        log("Force rebuild — rebuilding BM25 index...")
    else:
        log("No existing BM25 index — building...")

    bm25.build_index(documents)
    bm25.save_index()
    stats = bm25.get_index_stats()
    log(f"✅ BM25 index built  ({time.time()-t2:.1f}s)")
    log(f"   Documents : {stats.get('total_documents', 0):,}")
    log(f"   Vocab     : {stats.get('vocabulary_size', 0):,}")


# ──────────────────────────────────────────────────────────
# FINAL STATUS
# ──────────────────────────────────────────────────────────
step("FINAL STATUS")

qdrant_info = vector_store.get_collection_info()
bm25_stats  = bm25.get_index_stats()

log(f"🔷 Qdrant Vector Store")
log(f"   Collection : {Config.COLLECTION_NAME}")
log(f"   Vectors    : {qdrant_info.get('points_count', 0):,}")

log(f"\n📊 BM25 Index")
log(f"   Documents  : {bm25_stats.get('total_documents', 0):,}")
log(f"   Vocab size : {bm25_stats.get('vocabulary_size', 0):,}")
log(f"   Avg length : {bm25_stats.get('average_document_length', 0):.1f} tokens")

log(f"\n✅ RAG indices ready! Total time: {time.time()-t0:.1f}s")
log("   You can now run: streamlit run app.py")
