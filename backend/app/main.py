"""
FastAPI application for AI-SME system.
Main entry point with lifespan management.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import init_db, close_db, check_db_connection, check_extensions
from app.core.memory import get_memory_manager
from app.agents.registry import get_agent_registry
from app.orchestration.main_graph import create_main_graph

# Initialize logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown logic.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("AI-SME System Starting...")
    logger.info("=" * 80)

    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_db()

        # Check database connection
        db_ok = await check_db_connection()
        if not db_ok:
            logger.error("Database connection failed!")
            raise RuntimeError("Database connection failed")

        # Check extensions
        extensions = await check_extensions()
        missing = [ext for ext, installed in extensions.items() if not installed]
        if missing:
            logger.error("Missing PostgreSQL extensions", extensions=missing)
            raise RuntimeError(f"Missing extensions: {missing}")

        # Initialize memory system (Phase 2)
        logger.info("Initializing memory system...")
        memory_manager = get_memory_manager()
        await memory_manager.initialize()
        app.state.memory = memory_manager

        # Initialize agent registry (Phase 2)
        logger.info("Initializing agent registry...")
        agent_registry = get_agent_registry()

        # Initialize all agents that don't require corpus
        agent_instances = agent_registry.initialize_all_agents(lazy=True)
        app.state.agent_registry = agent_registry
        app.state.agent_instances = agent_instances

        # Initialize main orchestration graph (Phase 2)
        logger.info("Initializing main orchestration graph...")
        main_graph = create_main_graph(
            agent_registry=agent_instances,
            enable_query_refinement=True,
            enable_critique=True,
            critique_threshold=0.7
        )
        app.state.main_graph = main_graph

        logger.info("=" * 80)
        logger.info("AI-SME System Ready! (Phase 2)")
        logger.info(f"Agents available: {', '.join(agent_instances.keys())}")
        logger.info(f"Memory system: enabled with checkpointing")
        logger.info(f"Main graph: query refinement + routing + critique")
        logger.info(f"API Host: {settings.api_host}:{settings.api_port}")
        logger.info(f"CORS Origins: {settings.cors_origins}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("Startup failed", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("=" * 80)
    logger.info("AI-SME System Shutting Down...")
    logger.info("=" * 80)

    try:
        # Shutdown memory system
        if hasattr(app.state, "memory"):
            await app.state.memory.shutdown()
            logger.info("Memory system shutdown")

        # Close database connections
        await close_db()
        logger.info("Database connections closed")

        logger.info("Shutdown complete")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("Shutdown error", error=str(e))


# Create FastAPI app
app = FastAPI(
    title="AI-SME System API",
    description="Multi-agent AI consultant with domain-aware reasoning and hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI-SME System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database
        db_ok = await check_db_connection()

        return {
            "status": "healthy" if db_ok else "unhealthy",
            "database": "connected" if db_ok else "disconnected",
            "version": "1.0.0"
        }

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Import and include routers
from app.api.chat import router as chat_router
from app.api.agents import router as agents_router

app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(agents_router, prefix="/api", tags=["agents"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,  # For development
        log_level=settings.log_level.lower()
    )
