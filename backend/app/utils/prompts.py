"""
Prompt templates for the AI-SME system.
Hardcoded prompts following the requirements (Phase 1).
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ============================================================================
# SYNTHESIS PROMPTS
# ============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are an expert AI consultant specialized in providing accurate, evidence-based answers.

Your task is to synthesize information from the provided context to answer the user's question.

## Guidelines:
1. **Accuracy First**: Only use information present in the context. Do NOT hallucinate or make up information.
2. **Citations**: Reference specific sources using [1], [2], etc. format, corresponding to the document numbers in the context.
3. **Clarity**: Provide clear, concise answers in professional language.
4. **Structure**: Use markdown formatting for better readability.
5. **Completeness**: Address all aspects of the question if possible from the context.
6. **Uncertainty**: If the context doesn't contain enough information, explicitly state what is missing.

## Output Format:
- Start with a direct answer to the question
- Support claims with specific references [1], [2], etc.
- Include relevant details and examples from the context
- End with a confidence statement if appropriate

Remember: ONLY cite information actually present in the provided context. Never make up citations."""

SYNTHESIS_USER_PROMPT = """Question: {query}

Context:
{context}

Please provide a comprehensive answer based on the context above, using [N] notation to reference specific sources."""


def get_synthesis_prompt() -> ChatPromptTemplate:
    """
    Get the synthesis prompt template for generating answers.

    Returns:
        ChatPromptTemplate for synthesis
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_SYSTEM_PROMPT),
        ("user", SYNTHESIS_USER_PROMPT),
    ])


# ============================================================================
# QUERY REFINEMENT PROMPTS (for Phase 2+)
# ============================================================================

QUERY_REFINEMENT_PROMPT = """Analyze and refine the following user query for better retrieval.

Original Query: {query}

Tasks:
1. Disambiguate any unclear terms
2. Identify key topical facets
3. Suggest specific search intents
4. Rephrase for optimal retrieval

Output a JSON with:
{{
    "refined_query": "...",
    "facets": ["facet1", "facet2"],
    "search_intents": ["intent1", "intent2"]
}}"""


# ============================================================================
# REFLECTION PROMPTS (for SelfRAGAgent, Phase 3)
# ============================================================================

REFLECTION_PROMPT = """Review the retrieved context for the query and identify gaps.

Query: {query}

Retrieved Context: {context}

Questions to consider:
1. What key information is missing?
2. Are there unanswered aspects of the query?
3. What additional topics should be explored?

Output a JSON with:
{{
    "missing_topics": ["topic1", "topic2"],
    "confidence": 0.0-1.0,
    "needs_more_retrieval": true/false,
    "suggested_searches": ["search1", "search2"]
}}"""


# ============================================================================
# VALIDATION PROMPTS (for Neuro-Symbolic Validator, Phase 3)
# ============================================================================

VALIDATION_PROMPT = """Validate the following answer against the source context.

Query: {query}

Answer: {answer}

Source Context: {context}

Evaluate:
1. Factuality: Are all claims supported by the context?
2. Consistency: Are there any contradictions?
3. Completeness: Does it address all aspects of the query?

Output a JSON with scores (0.0-1.0):
{{
    "factuality": 0.0-1.0,
    "consistency": 0.0-1.0,
    "completeness": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "low_confidence_claims": ["claim1", "claim2"]
}}"""


# ============================================================================
# CRITIQUE PROMPTS (for CritiqueAgent, Phase 2+)
# ============================================================================

CRITIQUE_PROMPT = """Critique the following answer for quality and completeness.

Query: {query}

Answer: {answer}

Review for:
1. Completeness: Are all aspects addressed?
2. Alternative views: Are there other perspectives?
3. Contradictions: Any inconsistencies?
4. Bias: Any potential biases?

Provide constructive feedback and specific improvements."""


# ============================================================================
# PLAN DECOMPOSITION PROMPT (for PlanThenReadAgent, Phase 2)
# ============================================================================

PLAN_DECOMPOSITION_PROMPT = """Break down the following complex question into simpler sub-questions.

Question: {query}

Decompose into:
1. Independent sub-questions that can be answered separately
2. Logical sequence if dependencies exist
3. Clear, specific sub-questions

Output a JSON array:
[
    {{"sub_question": "...", "reasoning": "..."}},
    ...
]"""


# ============================================================================
# EXTRACTION PROMPTS (for EvidenceFirstAgent, Phase 2)
# ============================================================================

EXTRACTION_PROMPT = """Extract exact spans from the context that answer the query.

Query: {query}

Context: {context}

Extract:
1. Exact text spans (no paraphrasing)
2. Source reference for each span
3. Relevance score for each span

Output JSON:
[
    {{
        "span": "exact text",
        "source_id": "doc_id",
        "relevance": 0.0-1.0,
        "reasoning": "why this span is relevant"
    }},
    ...
]"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_context_with_citations(docs: list, start_index: int = 0) -> str:
    """
    Format a list of documents with citation numbers.

    Args:
        docs: List of Document objects
        start_index: Starting citation number

    Returns:
        Formatted string with [N] citations
    """
    formatted_parts = []
    for i, doc in enumerate(docs, start=start_index):
        citation_num = i + 1
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}

        # Include source info if available
        source_info = ""
        if "title" in metadata:
            source_info = f" (Source: {metadata['title']})"
        elif "filename" in metadata:
            source_info = f" (Source: {metadata['filename']})"

        formatted_parts.append(f"[{citation_num}]{source_info}\n{content}")

    return "\n\n".join(formatted_parts)


def extract_citations_from_answer(answer: str, context_docs: list) -> list[dict]:
    """
    Extract citation references from answer and map to source documents.

    Args:
        answer: Generated answer with [N] citations
        context_docs: List of source documents

    Returns:
        List of citation dictionaries
    """
    import re

    citations = []
    citation_pattern = r'\[(\d+)\]'
    cited_numbers = set(map(int, re.findall(citation_pattern, answer)))

    for num in sorted(cited_numbers):
        idx = num - 1
        if 0 <= idx < len(context_docs):
            doc = context_docs[idx]
            citations.append({
                "citation_number": num,
                "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                "doc_id": doc.metadata.get("id", "") if hasattr(doc, 'metadata') else ""
            })

    return citations
