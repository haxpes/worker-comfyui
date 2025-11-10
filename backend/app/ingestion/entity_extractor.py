"""
Entity Extractor: Extract named entities from chunks using spaCy and LLM.
Supports custom entity types for technical domains.
"""
import asyncio
from typing import List, Dict, Set, Optional, Tuple
from uuid import UUID
import spacy
from spacy.tokens import Doc
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import Chunk, Entity, EntityOccurrence
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger
from sqlalchemy import select, delete
import json

logger = get_logger(__name__)

# Entity type mapping
STANDARD_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT",
    "EVENT", "LAW", "LANGUAGE", "DATE", "TIME",
    "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
}

CUSTOM_TYPES = {
    "STANDARD",  # Technical standards (ISO, IEEE, etc.)
    "REGULATION",  # Regulatory requirements
    "ARTIFACT",  # Code artifacts, frameworks, tools
    "CONCEPT",  # Technical concepts
    "METHODOLOGY"  # Processes, methodologies
}

ENTITY_ENHANCEMENT_PROMPT = """You are an expert at enriching entity information with concise descriptions.

Entity: {entity_name}
Type: {entity_type}
Context: {context}

Task: Provide a brief 1-2 sentence description of this entity based on the context.
Focus on:
1. What the entity is
2. Its role or significance in the context

Output only the description, nothing else."""


class EntityExtractor:
    """
    Extract and enrich entities from text chunks.
    Uses spaCy NER + LLM enhancement + custom rules.
    """

    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize entity extractor.

        Args:
            model_name: spaCy model name
        """
        logger.info(f"Loading spaCy model: {model_name}")

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Attempting download...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )

        self.embedder = get_embedder()

        logger.info("EntityExtractor initialized")

    async def extract_from_corpus(
        self,
        corpus_id: str,
        force_rebuild: bool = False,
        enhance_with_llm: bool = True,
        batch_size: int = 50
    ) -> Dict[str, int]:
        """
        Extract entities from all chunks in a corpus.

        Args:
            corpus_id: Corpus ID
            force_rebuild: Delete existing entities first
            enhance_with_llm: Use LLM to enrich entity descriptions
            batch_size: Chunks to process per batch

        Returns:
            Statistics dictionary
        """
        logger.info(
            "Starting entity extraction",
            corpus_id=corpus_id,
            force_rebuild=force_rebuild,
            enhance_with_llm=enhance_with_llm
        )

        async with get_db_session() as session:
            # Delete existing entities if force rebuild
            if force_rebuild:
                logger.info("Force rebuild: deleting existing entities")
                await session.execute(
                    delete(Entity).where(Entity.corpus_id == corpus_id)
                )
                await session.commit()

            # Get all chunks for corpus
            result = await session.execute(
                select(Chunk).where(Chunk.corpus_id == corpus_id)
            )
            chunks = result.scalars().all()

            if not chunks:
                logger.warning(f"No chunks found for corpus {corpus_id}")
                return {"entities": 0, "occurrences": 0}

            logger.info(f"Processing {len(chunks)} chunks")

        # Extract entities from chunks in batches
        all_entities_map = {}  # name+type -> entity dict
        all_occurrences = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

            batch_entities, batch_occurrences = await self._extract_batch(
                batch=batch,
                corpus_id=corpus_id,
                enhance_with_llm=enhance_with_llm
            )

            # Merge entities (consolidate duplicates)
            for entity_key, entity_data in batch_entities.items():
                if entity_key in all_entities_map:
                    # Update confidence (take max)
                    all_entities_map[entity_key]["confidence"] = max(
                        all_entities_map[entity_key]["confidence"],
                        entity_data["confidence"]
                    )
                else:
                    all_entities_map[entity_key] = entity_data

            all_occurrences.extend(batch_occurrences)

        # Generate embeddings for entities
        logger.info("Generating entity embeddings...")
        entity_texts = [
            f"{data['name']}: {data.get('description', '')}"
            for data in all_entities_map.values()
        ]

        embeddings = await self.embedder.embed_batch(entity_texts)

        # Add embeddings to entities
        for (entity_key, embedding) in zip(all_entities_map.keys(), embeddings):
            all_entities_map[entity_key]["embedding"] = embedding

        # Store entities and occurrences
        logger.info("Storing entities in database...")
        entity_id_map = {}  # entity_key -> UUID

        async with get_db_session() as session:
            # Insert entities
            for entity_key, entity_data in all_entities_map.items():
                entity = Entity(
                    corpus_id=corpus_id,
                    name=entity_data["name"],
                    entity_type=entity_data["type"],
                    description=entity_data.get("description"),
                    embedding=entity_data.get("embedding"),
                    confidence=entity_data["confidence"],
                    metadata=entity_data.get("metadata", {})
                )
                session.add(entity)
                await session.flush()  # Get ID
                entity_id_map[entity_key] = entity.id

            await session.commit()

            # Insert occurrences
            for occurrence_data in all_occurrences:
                entity_key = occurrence_data["entity_key"]
                entity_id = entity_id_map.get(entity_key)

                if not entity_id:
                    continue

                occurrence = EntityOccurrence(
                    entity_id=entity_id,
                    chunk_id=occurrence_data["chunk_id"],
                    position=occurrence_data.get("position"),
                    context=occurrence_data.get("context")
                )
                session.add(occurrence)

            await session.commit()

        stats = {
            "entities": len(all_entities_map),
            "occurrences": len(all_occurrences),
            "chunks_processed": len(chunks)
        }

        logger.info(
            "Entity extraction completed",
            **stats
        )

        return stats

    async def _extract_batch(
        self,
        batch: List[Chunk],
        corpus_id: str,
        enhance_with_llm: bool
    ) -> Tuple[Dict[str, Dict], List[Dict]]:
        """
        Extract entities from a batch of chunks.

        Args:
            batch: List of chunks
            corpus_id: Corpus ID
            enhance_with_llm: Use LLM enhancement

        Returns:
            Tuple of (entities_map, occurrences_list)
        """
        entities_map = {}
        occurrences = []

        # Process chunks with spaCy
        texts = [chunk.content for chunk in batch]
        docs = list(self.nlp.pipe(texts))

        for chunk, doc in zip(batch, docs):
            chunk_entities = self._extract_from_doc(doc, chunk.id)

            for entity_data in chunk_entities:
                entity_key = f"{entity_data['name']}::{entity_data['type']}"

                # Add to entities map (first occurrence wins for description)
                if entity_key not in entities_map:
                    entities_map[entity_key] = entity_data

                # Add occurrence
                occurrences.append({
                    "entity_key": entity_key,
                    "chunk_id": chunk.id,
                    "position": entity_data.get("position"),
                    "context": entity_data.get("context")
                })

        # Enhance with LLM (sample for performance)
        if enhance_with_llm and entities_map:
            # Enhance up to 10 entities per batch
            sample_keys = list(entities_map.keys())[:10]

            for entity_key in sample_keys:
                entity_data = entities_map[entity_key]

                # Find a good context (first occurrence)
                context_occurrence = next(
                    (occ for occ in occurrences if occ["entity_key"] == entity_key),
                    None
                )

                if context_occurrence:
                    enhanced_desc = await self._enhance_entity(
                        name=entity_data["name"],
                        entity_type=entity_data["type"],
                        context=context_occurrence.get("context", "")
                    )

                    if enhanced_desc:
                        entities_map[entity_key]["description"] = enhanced_desc

        return entities_map, occurrences

    def _extract_from_doc(self, doc: Doc, chunk_id: UUID) -> List[Dict]:
        """
        Extract entities from a spaCy Doc.

        Args:
            doc: spaCy Doc
            chunk_id: Chunk ID

        Returns:
            List of entity dictionaries
        """
        entities = []
        seen = set()

        for ent in doc.ents:
            # Skip if already seen
            entity_key = f"{ent.text.lower()}::{ent.label_}"
            if entity_key in seen:
                continue

            seen.add(entity_key)

            # Map to entity type
            entity_type = self._map_entity_type(ent.label_)

            # Get context (surrounding text)
            start_idx = max(0, ent.start_char - 100)
            end_idx = min(len(doc.text), ent.end_char + 100)
            context = doc.text[start_idx:end_idx]

            entities.append({
                "name": ent.text,
                "type": entity_type,
                "position": ent.start_char,
                "context": context,
                "confidence": 0.9,  # spaCy confidence proxy
                "metadata": {
                    "spacy_label": ent.label_
                }
            })

        return entities

    def _map_entity_type(self, spacy_label: str) -> str:
        """
        Map spaCy label to our entity type system.

        Args:
            spacy_label: spaCy NER label

        Returns:
            Mapped entity type
        """
        # Check if standard type
        if spacy_label in STANDARD_TYPES:
            return spacy_label

        # Custom heuristics for technical entities
        if "ISO" in spacy_label or "IEEE" in spacy_label:
            return "STANDARD"

        if "REG" in spacy_label or "CFR" in spacy_label:
            return "REGULATION"

        # Default to the spacy label
        return spacy_label

    async def _enhance_entity(
        self,
        name: str,
        entity_type: str,
        context: str
    ) -> Optional[str]:
        """
        Enhance entity with LLM-generated description.

        Args:
            name: Entity name
            entity_type: Entity type
            context: Surrounding context

        Returns:
            Enhanced description or None
        """
        try:
            prompt = ChatPromptTemplate.from_template(ENTITY_ENHANCEMENT_PROMPT)
            chain = prompt | self.llm | StrOutputParser()

            description = await chain.ainvoke({
                "entity_name": name,
                "entity_type": entity_type,
                "context": context[:500]  # Limit context length
            })

            return description.strip()

        except Exception as e:
            logger.warning(
                "Entity enhancement failed",
                entity=name,
                error=str(e)
            )
            return None


def create_entity_extractor() -> EntityExtractor:
    """
    Create an EntityExtractor instance.

    Returns:
        EntityExtractor
    """
    return EntityExtractor()
