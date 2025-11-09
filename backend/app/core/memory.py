"""
Memory management with LangGraph integration.
Uses PostgresSaver for checkpointing and PostgresStore for long-term memory.
"""
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import BaseMessage
from app.config import settings
from app.db.connection import get_db_session
from app.db.models import UserPreference, ConversationSummary
from app.ingestion.embedder import get_embedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Unified memory management for the AI-SME system.
    Integrates LangGraph checkpointing and custom memory storage.
    """

    def __init__(self):
        """Initialize memory manager."""
        self.checkpointer: Optional[AsyncPostgresSaver] = None
        self.embedder = get_embedder()
        logger.info("MemoryManager initialized")

    async def initialize(self):
        """
        Initialize memory components.
        Sets up LangGraph checkpointer.
        """
        try:
            # Initialize PostgresSaver for LangGraph checkpointing
            # Note: LangGraph will create its own tables (checkpoints, writes)
            self.checkpointer = AsyncPostgresSaver.from_conn_string(
                settings.database_url
            )
            await self.checkpointer.setup()

            logger.info("Memory system initialized", checkpointer=True)

        except Exception as e:
            logger.error("Failed to initialize memory system", error=str(e))
            raise

    async def shutdown(self):
        """Cleanup memory connections."""
        try:
            if self.checkpointer:
                # LangGraph checkpointer doesn't need explicit cleanup in newer versions
                logger.info("Memory system shutdown")

        except Exception as e:
            logger.error("Error during memory shutdown", error=str(e))

    # ========================================================================
    # CONVERSATION MEMORY
    # ========================================================================

    async def save_conversation_summary(
        self,
        thread_id: str,
        user_id: Optional[str],
        messages: List[BaseMessage],
        summary_text: str
    ) -> None:
        """
        Save conversation summary with embedding.

        Args:
            thread_id: Thread/conversation ID
            user_id: Optional user ID
            messages: List of messages
            summary_text: Generated summary
        """
        try:
            # Generate embedding
            embedding = await self.embedder.embed_text(summary_text)

            async with get_db_session() as session:
                # Check if summary exists
                existing = await session.get(ConversationSummary, thread_id)

                if existing:
                    # Update existing
                    existing.summary = summary_text
                    existing.embedding = embedding
                    existing.message_count = len(messages)
                else:
                    # Create new
                    summary = ConversationSummary(
                        thread_id=thread_id,
                        user_id=user_id,
                        summary=summary_text,
                        embedding=embedding,
                        message_count=len(messages)
                    )
                    session.add(summary)

                await session.commit()

            logger.info(
                "Conversation summary saved",
                thread_id=thread_id,
                message_count=len(messages)
            )

        except Exception as e:
            logger.error(
                "Failed to save conversation summary",
                thread_id=thread_id,
                error=str(e)
            )

    async def get_conversation_summary(
        self,
        thread_id: str
    ) -> Optional[str]:
        """
        Get conversation summary for a thread.

        Args:
            thread_id: Thread ID

        Returns:
            Summary text or None
        """
        try:
            async with get_db_session() as session:
                summary = await session.get(ConversationSummary, thread_id)
                return summary.summary if summary else None

        except Exception as e:
            logger.error(
                "Failed to get conversation summary",
                thread_id=thread_id,
                error=str(e)
            )
            return None

    async def search_similar_conversations(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar past conversations using embeddings.

        Args:
            query: Search query
            user_id: Optional user ID filter
            limit: Max results

        Returns:
            List of similar conversation summaries
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedder.embed_text(query)

            async with get_db_session() as session:
                from sqlalchemy import select

                stmt = (
                    select(
                        ConversationSummary.thread_id,
                        ConversationSummary.summary,
                        ConversationSummary.message_count,
                        ConversationSummary.embedding.cosine_distance(query_embedding).label('distance')
                    )
                    .where(ConversationSummary.embedding.isnot(None))
                )

                if user_id:
                    stmt = stmt.where(ConversationSummary.user_id == user_id)

                stmt = stmt.order_by('distance').limit(limit)

                result = await session.execute(stmt)
                rows = result.all()

                conversations = []
                for row in rows:
                    similarity = 1.0 - (float(row.distance) / 2.0)
                    conversations.append({
                        "thread_id": row.thread_id,
                        "summary": row.summary,
                        "message_count": row.message_count,
                        "similarity": similarity
                    })

                return conversations

        except Exception as e:
            logger.error("Failed to search conversations", error=str(e))
            return []

    # ========================================================================
    # USER PREFERENCES
    # ========================================================================

    async def save_user_preference(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> None:
        """
        Save user preferences.

        Args:
            user_id: User ID
            preferences: Preferences dictionary
        """
        try:
            async with get_db_session() as session:
                existing = await session.get(UserPreference, user_id)

                if existing:
                    existing.preferences = preferences
                else:
                    pref = UserPreference(
                        user_id=user_id,
                        preferences=preferences
                    )
                    session.add(pref)

                await session.commit()

            logger.info("User preferences saved", user_id=user_id)

        except Exception as e:
            logger.error(
                "Failed to save user preferences",
                user_id=user_id,
                error=str(e)
            )

    async def get_user_preferences(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get user preferences.

        Args:
            user_id: User ID

        Returns:
            Preferences dictionary
        """
        try:
            async with get_db_session() as session:
                pref = await session.get(UserPreference, user_id)
                return pref.preferences if pref else {}

        except Exception as e:
            logger.error(
                "Failed to get user preferences",
                user_id=user_id,
                error=str(e)
            )
            return {}

    async def update_query_patterns(
        self,
        user_id: str,
        query: str,
        selected_agents: List[str],
        success: bool
    ) -> None:
        """
        Update learned query patterns for router bias.

        Args:
            user_id: User ID
            query: Query text
            selected_agents: Agents that were selected
            success: Whether the interaction was successful
        """
        try:
            async with get_db_session() as session:
                pref = await session.get(UserPreference, user_id)

                if not pref:
                    pref = UserPreference(
                        user_id=user_id,
                        preferences={},
                        query_patterns={}
                    )
                    session.add(pref)

                # Update query patterns
                patterns = pref.query_patterns or {}

                # Simple pattern learning: track successful agent selections
                for agent in selected_agents:
                    if agent not in patterns:
                        patterns[agent] = {
                            "count": 0,
                            "success_count": 0,
                            "example_queries": []
                        }

                    patterns[agent]["count"] += 1
                    if success:
                        patterns[agent]["success_count"] += 1

                    # Store a few example queries
                    if len(patterns[agent]["example_queries"]) < 5:
                        patterns[agent]["example_queries"].append(query[:100])

                pref.query_patterns = patterns
                await session.commit()

            logger.debug(
                "Query patterns updated",
                user_id=user_id,
                agents=selected_agents
            )

        except Exception as e:
            logger.error(
                "Failed to update query patterns",
                user_id=user_id,
                error=str(e)
            )

    async def get_query_patterns(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get learned query patterns for a user.

        Args:
            user_id: User ID

        Returns:
            Query patterns dictionary
        """
        try:
            async with get_db_session() as session:
                pref = await session.get(UserPreference, user_id)
                return pref.query_patterns if pref else {}

        except Exception as e:
            logger.error(
                "Failed to get query patterns",
                user_id=user_id,
                error=str(e)
            )
            return {}

    # ========================================================================
    # SUMMARY GENERATION
    # ========================================================================

    async def should_create_summary(
        self,
        thread_id: str,
        message_count: int
    ) -> bool:
        """
        Determine if a conversation summary should be created.

        Args:
            thread_id: Thread ID
            message_count: Current message count

        Returns:
            True if summary should be created
        """
        # Create summary every N messages
        interval = settings.conversation_summary_interval

        # Check if we've crossed a summary boundary
        if message_count % interval == 0:
            return True

        return False

    async def generate_conversation_summary(
        self,
        messages: List[BaseMessage]
    ) -> str:
        """
        Generate a conversation summary from messages.

        Args:
            messages: List of messages

        Returns:
            Summary text
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Simple summarization for now
        # TODO: Use more sophisticated summarization in production
        try:
            # Combine messages
            conversation = "\n\n".join([
                f"{msg.type}: {msg.content}" for msg in messages[-10:]  # Last 10 messages
            ])

            prompt = ChatPromptTemplate.from_template(
                "Summarize the following conversation concisely (2-3 sentences):\n\n{conversation}"
            )

            llm = ChatOpenAI(
                model=settings.default_model,
                temperature=0,
                openai_api_key=settings.openai_api_key
            )

            chain = prompt | llm | StrOutputParser()
            summary = await chain.ainvoke({"conversation": conversation})

            return summary

        except Exception as e:
            logger.error("Failed to generate conversation summary", error=str(e))
            # Fallback: simple concatenation
            return f"Conversation with {len(messages)} messages"


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """
    Get global memory manager instance.

    Returns:
        MemoryManager instance
    """
    global _memory_manager

    if _memory_manager is None:
        _memory_manager = MemoryManager()

    return _memory_manager


@asynccontextmanager
async def memory_lifespan():
    """
    Context manager for memory lifespan.
    Use in FastAPI lifespan.
    """
    manager = get_memory_manager()

    # Startup
    await manager.initialize()

    try:
        yield manager
    finally:
        # Shutdown
        await manager.shutdown()
