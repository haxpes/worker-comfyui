"""
Database connection management with connection pooling.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncpg
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Global engine and session factory
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the global database engine.

    Returns:
        AsyncEngine instance
    """
    global _engine

    if _engine is None:
        # Convert postgresql:// to postgresql+asyncpg://
        db_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

        logger.info(
            "Creating database engine",
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )

        _engine = create_async_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            echo=settings.log_level == "DEBUG",  # SQL logging in debug mode
            future=True,
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the global session factory.

    Returns:
        Session factory
    """
    global _session_factory

    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session (context manager).

    Usage:
        async with get_db_session() as session:
            result = await session.execute(...)

    Yields:
        AsyncSession instance
    """
    factory = get_session_factory()
    session = factory()

    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """
    Initialize database: create tables and extensions.
    Executes init_db.sql script.
    """
    from pathlib import Path

    logger.info("Initializing database schema...")

    # Read SQL script
    sql_file = Path(__file__).parent / "init_db.sql"
    if not sql_file.exists():
        logger.error("init_db.sql not found", path=str(sql_file))
        raise FileNotFoundError(f"SQL init script not found: {sql_file}")

    sql_script = sql_file.read_text()

    # Execute using raw asyncpg connection (for CREATE EXTENSION support)
    db_url = settings.database_url.replace("postgresql://", "")  # Remove scheme
    parts = db_url.split("@")
    user_pass = parts[0].split(":")
    host_db = parts[1].split("/")
    host_port = host_db[0].split(":")

    user = user_pass[0]
    password = user_pass[1] if len(user_pass) > 1 else ""
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else 5432
    database = host_db[1] if len(host_db) > 1 else "ai_sme"

    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )

        try:
            await conn.execute(sql_script)
            logger.info("Database schema initialized successfully")
        finally:
            await conn.close()

    except Exception as e:
        logger.error(
            "Failed to initialize database",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise


async def close_db() -> None:
    """
    Close database connections and dispose of engine.
    """
    global _engine, _session_factory

    if _engine:
        logger.info("Closing database connections...")
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")


async def check_db_connection() -> bool:
    """
    Check if database connection is working.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        async with get_db_session() as session:
            result = await session.execute("SELECT 1")
            result.scalar()
            logger.info("Database connection check: OK")
            return True
    except Exception as e:
        logger.error("Database connection check failed", error=str(e))
        return False


async def check_extensions() -> dict[str, bool]:
    """
    Check if required PostgreSQL extensions are installed.

    Returns:
        Dictionary mapping extension name to availability
    """
    required_extensions = ["uuid-ossp", "vector", "pg_trgm"]
    results = {}

    try:
        async with get_db_session() as session:
            for ext in required_extensions:
                query = f"SELECT COUNT(*) FROM pg_extension WHERE extname = '{ext}'"
                result = await session.execute(query)
                count = result.scalar()
                results[ext] = count > 0

                if count > 0:
                    logger.info(f"Extension '{ext}': installed")
                else:
                    logger.warning(f"Extension '{ext}': NOT installed")

        return results

    except Exception as e:
        logger.error("Failed to check extensions", error=str(e))
        return {ext: False for ext in required_extensions}
