"""External memory, vector stores, RAG, KV caches."""

from typing import Dict

from dsl import (  # noqa: F401
    Spec, _io, _op, _norm, _act, _attn, _merge, _emb, _loss, _ref, _notes,
)

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
        [],
        None,
        _notes(
            used_in=[
                "Neural Turing Machines / Differentiable Neural Computers",
                "Memory-Augmented Networks (MANN)",
                "Memorizing Transformers (Wu et al. 2022)",
            ],
            tasks=[
                "Few-shot learning where context is too large to keep in attention",
                "Long-term episodic memory in agents",
            ],
            pitfalls=[
                "Memory bank grows unbounded — needs eviction policy or summarisation.",
                "Soft addressing is slow to learn — hard / sparse attention often does better in practice "
                "but is non-differentiable.",
                "Mostly superseded by retrieval (RAG / kNN-LM) for language tasks.",
            ],
            see_also=[
                "[Neural Turing Machines (Graves et al. 2014)](https://arxiv.org/abs/1410.5401)",
                "[DNC (Graves et al. 2016)](https://www.nature.com/articles/nature20101)",
                "[Memorizing Transformers (Wu et al. 2022)](https://arxiv.org/abs/2203.08913)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "FAISS, ScaNN, Annoy — open-source ANN libraries",
                "Pinecone, Weaviate, Qdrant, Milvus — managed vector DBs",
                "OpenAI / Cohere / Voyage embedding APIs",
            ],
            tasks=[
                "Semantic search, question answering, recommendation",
                "Retrieval branch of RAG / kNN-LM / Memorising Transformers",
            ],
            pitfalls=[
                "Index parameters (HNSW M, efConstruction; IVF nlist; PQ nbits) need tuning per data scale — "
                "defaults rarely optimal.",
                "Embedding distribution drift between query and document encoders causes recall collapse.",
                "Cosine vs dot vs L2 — be explicit; mismatches between training loss and search metric "
                "silently degrade results.",
                "Re-indexing on embedding-model upgrade is expensive — version your indices.",
            ],
            see_also=[
                "[FAISS (Johnson et al. 2017)](https://arxiv.org/abs/1702.08734)",
                "[HNSW (Malkov & Yashunin 2016)](https://arxiv.org/abs/1603.09320)",
                "[ScaNN (Guo et al. 2020)](https://arxiv.org/abs/1908.10396)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Original RAG (Lewis et al. 2020) for open-domain QA",
                "REALM, RETRO — retrieval baked into pretraining",
                "ChatGPT-with-tools, Perplexity, Bing Copilot — production RAG",
                "LangChain / LlamaIndex / Haystack pipelines",
            ],
            tasks=[
                "Open-domain question answering with up-to-date knowledge",
                "Reducing hallucination in LLMs by grounding in retrieved documents",
                "Domain-specific assistants (legal, medical, code)",
            ],
            pitfalls=[
                "Retrieval quality dominates downstream LM quality — invest in the encoder / reranker.",
                "Top-k too large dilutes attention; too small misses the answer.",
                "Stale embedding model + fresh documents = recall collapse — re-embed on drift.",
                "Chunking strategy (sentence vs paragraph vs sliding window) materially changes recall.",
            ],
            see_also=[
                "[RAG (Lewis et al. 2020)](https://arxiv.org/abs/2005.11401)",
                "[REALM (Guu et al. 2020)](https://arxiv.org/abs/2002.08909)",
                "[RETRO (Borgeaud et al. 2021)](https://arxiv.org/abs/2112.04426)",
            ],
        ),
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
        [],
        None,
        _notes(
            used_in=[
                "Every production LLM inference stack (vLLM, TGI, TensorRT-LLM, llama.cpp)",
                "PagedAttention / vLLM's blocked KV cache",
                "Speculative decoding pipelines",
            ],
            tasks=[
                "Reducing autoregressive inference from O(T²) to O(T) per token",
                "Batched serving across multiple concurrent sessions",
            ],
            pitfalls=[
                "Memory is the bottleneck — `L × B × T × 2 × D` in fp16 is huge for long contexts.",
                "Multi-Query / Grouped-Query Attention shrink the cache (1 or g K-V heads instead of H).",
                "Beam search and continuous-batching need careful cache-block layout (see PagedAttention).",
                "Quantising the cache (FP8 / INT8) saves memory but can hurt long-context accuracy.",
            ],
            see_also=[
                "[Multi-Query Attention (Shazeer 2019)](https://arxiv.org/abs/1911.02150)",
                "[Grouped-Query Attention (Ainslie et al. 2023)](https://arxiv.org/abs/2305.13245)",
                "[PagedAttention / vLLM (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180)",
            ],
        ),
    ),
}
