"""
Structured logging configuration for the AI-SME system.
Supports both JSON and text formats with console and file handlers.
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any
import structlog
from app.config import settings


def setup_logging() -> structlog.BoundLogger:
    """
    Configure structured logging with console and file handlers.
    Resets log file on application restart.

    Returns:
        Configured structlog logger
    """
    # Ensure log directory exists
    log_file = Path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Reset log file on restart (overwrite mode)
    if log_file.exists():
        log_file.unlink()

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        handlers=[],  # We'll add handlers manually
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    # Format based on configuration
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
        console_formatter = logging.Formatter("%(message)s")
        file_formatter = logging.Formatter("%(message)s")
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_formatter = console_formatter

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, settings.log_level.upper()))
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Create and return logger
    logger = structlog.get_logger()
    logger.info(
        "Logging initialized",
        log_level=settings.log_level,
        log_file=settings.log_file,
        log_format=settings.log_format
    )

    return logger


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a logger instance with optional name binding.

    Args:
        name: Logger name (typically module name)

    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger


# Example usage patterns
def log_request(logger: structlog.BoundLogger, method: str, path: str, **kwargs: Any) -> None:
    """Log HTTP request with standard fields."""
    logger.info(
        "HTTP request",
        method=method,
        path=path,
        **kwargs
    )


def log_retrieval(
    logger: structlog.BoundLogger,
    retriever: str,
    query: str,
    results: int,
    duration_ms: float,
    **kwargs: Any
) -> None:
    """Log retrieval operation."""
    logger.info(
        "Retrieval completed",
        retriever=retriever,
        query_preview=query[:100],
        result_count=results,
        duration_ms=round(duration_ms, 2),
        **kwargs
    )


def log_agent_execution(
    logger: structlog.BoundLogger,
    agent: str,
    thread_id: str,
    duration_ms: float,
    success: bool,
    **kwargs: Any
) -> None:
    """Log agent execution."""
    logger.info(
        "Agent execution completed",
        agent=agent,
        thread_id=thread_id,
        duration_ms=round(duration_ms, 2),
        success=success,
        **kwargs
    )


def log_error(
    logger: structlog.BoundLogger,
    error: Exception,
    context: str,
    **kwargs: Any
) -> None:
    """Log error with context."""
    logger.error(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context,
        exc_info=True,
        **kwargs
    )
