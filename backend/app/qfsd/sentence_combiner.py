"""
Sentence Combination with spaCy Coherence.

Groups semantically similar sentences using FAISS and combines them
with linguistic coherence using spaCy.
"""
from typing import List, Optional
import spacy
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SentenceCombiner:
    """
    Sentence combiner using FAISS grouping + spaCy coherence.
    """

    def __init__(
        self,
        faiss_index: Optional = None,  # Optional[FAISSIndexBuilder]
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize sentence combiner.

        Args:
            faiss_index: Optional FAISS index for grouping
            spacy_model: spaCy model name
        """
        self.faiss_index = faiss_index

        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info("spaCy model loaded", model=spacy_model)
        except OSError:
            logger.warning(
                "spaCy model not found, downloading...",
                model=spacy_model
            )
            # Download model
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", spacy_model],
                check=True
            )
            self.nlp = spacy.load(spacy_model)
            logger.info("spaCy model downloaded and loaded")

        logger.info("Sentence combiner initialized")

    def semantic_grouping(
        self,
        sentences: List[str],
        similarity_threshold: float = 0.7
    ) -> List[List[str]]:
        """
        Group sentences by semantic similarity using FAISS.

        Args:
            sentences: List of sentences
            similarity_threshold: Similarity threshold for grouping

        Returns:
            List of sentence groups
        """
        if not self.faiss_index:
            # No FAISS, each sentence is its own group
            return [[s] for s in sentences]

        logger.info(
            "Grouping sentences",
            sentences=len(sentences),
            threshold=similarity_threshold
        )

        groups = []
        used = set()

        for i, sentence in enumerate(sentences):
            if i in used:
                continue

            # Find similar sentences
            similar = self.faiss_index.find_similar_sentences(
                sentence,
                k=len(sentences),
                similarity_threshold=similarity_threshold
            )

            # Group similar sentences
            group = [sentence]
            used.add(i)

            for similar_sent, similarity in similar:
                similar_idx = next(
                    (idx for idx, s in enumerate(sentences) if s == similar_sent),
                    -1
                )

                if similar_idx != -1 and similar_idx != i and similar_idx not in used:
                    group.append(similar_sent)
                    used.add(similar_idx)

            groups.append(group)

        logger.info(
            "Grouping completed",
            original=len(sentences),
            groups=len(groups)
        )

        return groups

    def combine_with_spacy(self, sentence_group: List[str]) -> str:
        """
        Combine sentences using spaCy for linguistic coherence.

        Args:
            sentence_group: Group of similar sentences

        Returns:
            Combined sentence
        """
        if len(sentence_group) == 1:
            return sentence_group[0]

        # Use spaCy to analyze each sentence
        docs = [self.nlp(s) for s in sentence_group]

        # Combine sentences with proper punctuation
        # (Simple version: just join with space)
        combined = " ".join(sentence_group)

        # Use spaCy to ensure proper sentence boundaries
        combined_doc = self.nlp(combined)

        # Return properly formatted text
        return combined_doc.text

    def combine_sentences(
        self,
        sentences: List[str],
        similarity_threshold: float = 0.7
    ) -> List[str]:
        """
        Complete sentence combination pipeline.

        Args:
            sentences: List of sentences
            similarity_threshold: Similarity threshold for grouping

        Returns:
            Combined sentences
        """
        logger.info(
            "Combining sentences",
            sentences=len(sentences)
        )

        # Group semantically similar sentences
        groups = self.semantic_grouping(sentences, similarity_threshold)

        # Combine each group with spaCy
        combined = []
        for group in groups:
            if len(group) == 1:
                combined.append(group[0])
            else:
                combined_sent = self.combine_with_spacy(group)
                combined.append(combined_sent)

        logger.info(
            "Combination completed",
            original=len(sentences),
            combined=len(combined)
        )

        return combined


def split_into_sentences(text: str, spacy_model: str = "en_core_web_sm") -> List[str]:
    """
    Split text into sentences using spaCy.

    Args:
        text: Text to split
        spacy_model: spaCy model name

    Returns:
        List of sentences
    """
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        # Download if not available
        import subprocess
        subprocess.run(
            ["python", "-m", "spacy", "download", spacy_model],
            check=True
        )
        nlp = spacy.load(spacy_model)

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    return sentences


def create_sentence_combiner(
    faiss_index: Optional = None,
    spacy_model: str = "en_core_web_sm"
) -> SentenceCombiner:
    """
    Create a sentence combiner instance.

    Args:
        faiss_index: Optional FAISS index
        spacy_model: spaCy model name

    Returns:
        SentenceCombiner
    """
    return SentenceCombiner(
        faiss_index=faiss_index,
        spacy_model=spacy_model
    )
