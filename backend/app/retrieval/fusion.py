"""
Document fusion and diversity algorithms.
Implements RRF (Reciprocal Rank Fusion) and MMR (Maximum Marginal Relevance).
"""
from typing import List, Set
import numpy as np
from langchain_core.documents import Document
from app.config import settings
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    doc_lists: List[List[Document]],
    k: int | None = None,
    c: int = 60
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) for combining ranked lists.

    Formula: RRF(d) = Σ(1 / (c + rank(d)))

    Args:
        doc_lists: List of ranked document lists from different retrievers
        k: Number of top documents to return
        c: Constant to prevent division by zero and reduce impact of top ranks (default 60)

    Returns:
        Fused and re-ranked document list
    """
    k = k or settings.fusion_k

    logger.debug(
        "Applying RRF fusion",
        num_lists=len(doc_lists),
        k=k,
        c=c
    )

    # Build score map: doc_id -> RRF score
    rrf_scores = {}
    doc_map = {}  # doc_id -> Document

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            doc_id = doc.metadata.get("id")
            if not doc_id:
                continue

            # RRF score contribution from this ranked list
            score = 1.0 / (c + rank)

            if doc_id in rrf_scores:
                rrf_scores[doc_id] += score
            else:
                rrf_scores[doc_id] = score
                doc_map[doc_id] = doc

    # Sort by RRF score (descending)
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Create result list
    fused_docs = []
    for doc_id in sorted_doc_ids[:k]:
        doc = doc_map[doc_id]
        # Add RRF score to metadata
        doc.metadata["rrf_score"] = rrf_scores[doc_id]
        fused_docs.append(doc)

    logger.info(
        "RRF fusion completed",
        input_docs=sum(len(lst) for lst in doc_lists),
        unique_docs=len(rrf_scores),
        output_docs=len(fused_docs)
    )

    return fused_docs


async def maximum_marginal_relevance(
    query: str,
    documents: List[Document],
    k: int | None = None,
    lambda_param: float | None = None
) -> List[Document]:
    """
    Maximum Marginal Relevance (MMR) for diversity.

    MMR = argmax[λ * sim(q, d) - (1-λ) * max(sim(d, d_i)) for d_i in selected]

    Args:
        query: Original query
        documents: Candidate documents
        k: Number of documents to select
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)

    Returns:
        Diversified document list
    """
    k = k or settings.mmr_k
    lambda_param = lambda_param if lambda_param is not None else settings.mmr_lambda

    if not documents:
        return []

    if len(documents) <= k:
        logger.debug("MMR: fewer documents than k, returning all")
        return documents

    logger.debug(
        "Applying MMR",
        num_docs=len(documents),
        k=k,
        lambda_param=lambda_param
    )

    embedder = get_embedder()

    # Get query embedding
    query_embedding = await embedder.embed_text(query)
    query_vec = np.array(query_embedding)

    # Get document embeddings (use existing if available, generate if not)
    doc_embeddings = []
    for doc in documents:
        # Check if embedding is in metadata
        if "embedding" in doc.metadata:
            doc_embeddings.append(np.array(doc.metadata["embedding"]))
        else:
            # Generate embedding
            emb = await embedder.embed_text(doc.page_content[:1000])  # Truncate for efficiency
            doc_embeddings.append(np.array(emb))

    doc_embeddings = np.array(doc_embeddings)

    # Calculate query-document similarities
    query_sims = np.dot(doc_embeddings, query_vec) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_vec)
    )

    # MMR selection
    selected_indices = []
    remaining_indices = set(range(len(documents)))

    for _ in range(k):
        if not remaining_indices:
            break

        mmr_scores = []
        for idx in remaining_indices:
            # Relevance component
            relevance = query_sims[idx]

            # Diversity component (max similarity to already selected docs)
            if selected_indices:
                selected_embeddings = doc_embeddings[selected_indices]
                doc_embedding = doc_embeddings[idx]

                similarities = np.dot(selected_embeddings, doc_embedding) / (
                    np.linalg.norm(selected_embeddings, axis=1) * np.linalg.norm(doc_embedding)
                )
                max_sim = np.max(similarities)
            else:
                max_sim = 0.0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((idx, mmr_score))

        # Select document with highest MMR score
        best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    # Build result list
    result = [documents[idx] for idx in selected_indices]

    logger.info(
        "MMR completed",
        input_docs=len(documents),
        output_docs=len(result),
        lambda_param=lambda_param
    )

    return result


async def apply_fusion_and_diversity(
    doc_lists: List[List[Document]],
    query: str,
    fusion_k: int | None = None,
    mmr_k: int | None = None,
    lambda_param: float | None = None
) -> List[Document]:
    """
    Apply both RRF fusion and MMR diversity in sequence.

    Args:
        doc_lists: Multiple ranked lists from different retrievers
        query: Original query
        fusion_k: Number of docs after RRF
        mmr_k: Number of docs after MMR
        lambda_param: MMR diversity parameter

    Returns:
        Fused and diversified document list
    """
    fusion_k = fusion_k or settings.fusion_k
    mmr_k = mmr_k or settings.mmr_k
    lambda_param = lambda_param if lambda_param is not None else settings.mmr_lambda

    logger.info(
        "Applying fusion and diversity",
        num_lists=len(doc_lists),
        fusion_k=fusion_k,
        mmr_k=mmr_k
    )

    # Step 1: RRF fusion
    fused_docs = reciprocal_rank_fusion(doc_lists, k=fusion_k)

    # Step 2: MMR diversity
    diverse_docs = await maximum_marginal_relevance(
        query=query,
        documents=fused_docs,
        k=mmr_k,
        lambda_param=lambda_param
    )

    logger.info(
        "Fusion and diversity completed",
        input_docs=sum(len(lst) for lst in doc_lists),
        after_fusion=len(fused_docs),
        final_docs=len(diverse_docs)
    )

    return diverse_docs
