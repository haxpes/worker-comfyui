"""
Chat API endpoints with streaming support.
"""
import json
import asyncio
import time
from typing import Any, AsyncGenerator
from uuid import uuid4
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Request/Response models
class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message", min_length=1)
    thread_id: str | None = Field(None, description="Thread/conversation ID")
    agent: str = Field("lean_hybrid", description="Agent to use")


class Citation(BaseModel):
    """Citation model."""
    citation_number: int
    content: str
    doc_id: str
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Chat response model (non-streaming)."""
    answer: str
    citations: list[Citation]
    confidence: float
    thread_id: str
    agent: str
    execution_time_ms: float


@router.post("/chat")
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    """
    Non-streaming chat endpoint.

    Args:
        request: Chat request
        req: FastAPI request object

    Returns:
        Chat response with answer and citations
    """
    thread_id = request.thread_id or str(uuid4())
    start_time = time.time()

    logger.info(
        "Chat request received",
        thread_id=thread_id,
        message_preview=request.message[:100],
        agent=request.agent
    )

    try:
        # Get agent
        if request.agent == "lean_hybrid":
            agent = req.app.state.lean_hybrid_agent
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent: {request.agent}"
            )

        # Run agent
        result = await agent.run(query=request.message)

        execution_time_ms = (time.time() - start_time) * 1000

        # Build response
        response = ChatResponse(
            answer=result.get("answer", ""),
            citations=[
                Citation(**citation) for citation in result.get("citations", [])
            ],
            confidence=result.get("confidence", 0.0),
            thread_id=thread_id,
            agent=request.agent,
            execution_time_ms=round(execution_time_ms, 2)
        )

        logger.info(
            "Chat request completed",
            thread_id=thread_id,
            agent=request.agent,
            execution_time_ms=round(execution_time_ms, 2),
            citations=len(response.citations)
        )

        return response

    except Exception as e:
        logger.error(
            "Chat request failed",
            thread_id=thread_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Args:
        request: Chat request
        req: FastAPI request object

    Returns:
        StreamingResponse with SSE events
    """
    thread_id = request.thread_id or str(uuid4())

    logger.info(
        "Chat stream request received",
        thread_id=thread_id,
        message_preview=request.message[:100],
        agent=request.agent
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        start_time = time.time()

        try:
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id, 'agent': request.agent})}\n\n"

            # Get agent
            if request.agent == "lean_hybrid":
                agent = req.app.state.lean_hybrid_agent
            else:
                error_msg = f"Unknown agent: {request.agent}"
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return

            # Get the graph
            graph = agent.get_graph()

            # Create initial state
            from app.core.state import create_initial_agent_state
            initial_state = create_initial_agent_state(request.message, agent.name)

            # Stream graph execution
            async for event in graph.astream(initial_state):
                # Send event for each node completion
                node_name = list(event.keys())[0] if event else "unknown"
                node_data = event.get(node_name, {})

                # Different event types based on node
                if node_name == "retrieve":
                    # After retrieval
                    doc_lists = node_data.get("retrieved_docs", [])
                    total_docs = sum(len(lst) for lst in doc_lists) if isinstance(doc_lists[0], list) else len(doc_lists)

                    yield f"data: {json.dumps({'type': 'retrieval', 'docs_retrieved': total_docs})}\n\n"

                elif node_name == "fuse":
                    # After fusion
                    fused_docs = node_data.get("retrieved_docs", [])
                    yield f"data: {json.dumps({'type': 'fusion', 'docs_fused': len(fused_docs)})}\n\n"

                elif node_name == "rerank":
                    # After reranking
                    reranked_docs = node_data.get("reranked_docs", [])
                    yield f"data: {json.dumps({'type': 'rerank', 'docs_reranked': len(reranked_docs)})}\n\n"

                elif node_name == "synthesize":
                    # After synthesis - send answer and citations
                    answer = node_data.get("answer", "")
                    citations = node_data.get("citations", [])
                    confidence = node_data.get("confidence", 0.0)

                    # Stream answer in chunks (simulate token streaming)
                    words = answer.split()
                    chunk_size = 5  # words per chunk

                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i+chunk_size])
                        if i + chunk_size < len(words):
                            chunk += " "

                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.05)  # Small delay for streaming effect

                    # Send citations
                    yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'confidence': confidence})}\n\n"

            # Send completion event
            execution_time_ms = (time.time() - start_time) * 1000
            yield f"data: {json.dumps({'type': 'done', 'execution_time_ms': round(execution_time_ms, 2)})}\n\n"

            logger.info(
                "Chat stream completed",
                thread_id=thread_id,
                execution_time_ms=round(execution_time_ms, 2)
            )

        except Exception as e:
            logger.error(
                "Chat stream failed",
                thread_id=thread_id,
                error=str(e),
                error_type=type(e).__name__
            )
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """
    Get conversation thread (placeholder for Phase 2).

    Args:
        thread_id: Thread ID

    Returns:
        Thread information
    """
    # TODO: Implement in Phase 2 with memory system
    return {
        "thread_id": thread_id,
        "messages": [],
        "note": "Memory system not yet implemented (Phase 2)"
    }
