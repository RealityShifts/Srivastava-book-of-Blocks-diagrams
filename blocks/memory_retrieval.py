"""External memory, vector stores, RAG, KV caches."""

from typing import Dict

from dsl import Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref  # noqa: F401

CATEGORY = "memory_retrieval"
CATEGORY_DESC = "External memory, vector stores, RAG, KV caches."

BLOCKS: Dict[str, Spec] = {
    "ExternalMemory": (
        "Soft attention over an external memory bank acts as a read operation.",
        "q:(B, D), mem:(M, D) → r:(B, D)",
        [
            [_io("query  (B, D)"), _io("memory bank  (M, D)")],
            [_attn("attention(query, mem, mem)")],
            [_io("retrieved  (B, D)")],
        ],
    ),
    "VectorStore": (
        "Encode a corpus into a vector index, then look up the top-k nearest neighbours.",
        "corpus → index;  q → top-k",
        [
            [_io("corpus  (N docs)")],
            [_op("encoder → embeddings")],
            [_op("ANN index  (FAISS / HNSW / IVFPQ)")],
            [_io("query → top-k  (k, D)")],
        ],
    ),
    "RAGModule": (
        "Retrieval-Augmented Generation: retrieve top-k docs, condition the LM on them.",
        "query → answer",
        [
            [_io("query  (T_q,)")],
            [_op("VectorStore.retrieve  top-k")],
            [_op("encode docs (token-level)")],
            [_attn("cross-attention (LM ← docs)")],
            [_io("generated answer  (T_a,)")],
        ],
    ),
    "KVCache": (
        "Cache K, V tensors per layer per token to skip re-encoding history.",
        "x_t → K_{1..t}, V_{1..t}",
        [
            [_io("x_t  (B, 1, D)")],
            [_op("compute K_t, V_t")],
            [_op("append to cache")],
            [_io("cached K, V  (L, B, T, D)")],
        ],
    ),
}
