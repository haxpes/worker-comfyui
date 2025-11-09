#!/usr/bin/env python3
"""
CLI tool for AI-SME system management.
Handles indexing, corpus management, and utilities.
"""
import asyncio
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.db.connection import init_db, check_db_connection, check_extensions
from app.ingestion.indexer import DocumentIndexer, index_corpus

# Initialize
setup_logging()
logger = get_logger(__name__)
console = Console()
app = typer.Typer(help="AI-SME System CLI")


@app.command()
def index(
    corpus_path: Path = typer.Argument(..., help="Path to corpus directory"),
    name: str = typer.Option(None, "--name", "-n", help="Corpus name (defaults to directory name)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reindex (delete existing corpus)"),
):
    """
    Index a corpus directory.

    Parses PDF files, chunks them, generates embeddings, and stores in database.
    """
    console.print(f"\n[bold blue]AI-SME Indexing Tool[/bold blue]\n")

    # Validate path
    if not corpus_path.exists():
        console.print(f"[red]Error: Path not found: {corpus_path}[/red]")
        raise typer.Exit(1)

    if not corpus_path.is_dir():
        console.print(f"[red]Error: Path is not a directory: {corpus_path}[/red]")
        raise typer.Exit(1)

    # Run indexing
    async def run_indexing():
        try:
            # Initialize database
            with console.status("[bold green]Initializing database...[/bold green]"):
                await init_db()
                db_ok = await check_db_connection()

                if not db_ok:
                    console.print("[red]Database connection failed![/red]")
                    raise typer.Exit(1)

                console.print("[green]✓[/green] Database connected")

            # Check extensions
            extensions = await check_extensions()
            missing = [ext for ext, ok in extensions.items() if not ok]
            if missing:
                console.print(f"[red]Missing extensions: {missing}[/red]")
                raise typer.Exit(1)

            console.print("[green]✓[/green] Extensions loaded\n")

            # Index corpus
            console.print(f"[bold]Indexing corpus:[/bold] {corpus_path}")
            console.print(f"[bold]Corpus name:[/bold] {name or corpus_path.name}")
            console.print(f"[bold]Force reindex:[/bold] {force}\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Indexing...", total=None)

                stats = await index_corpus(
                    corpus_path=corpus_path,
                    corpus_name=name,
                    force_reindex=force
                )

                progress.stop()

            # Display results
            console.print("\n[bold green]Indexing completed![/bold green]\n")

            table = Table(title="Indexing Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Corpus ID", stats["corpus_id"])
            table.add_row("Corpus Name", stats["corpus_name"])
            table.add_row("Documents Indexed", str(stats["documents_indexed"]))
            table.add_row("Chunks Created", str(stats["chunks_created"]))
            table.add_row("Duration", f"{stats['duration_seconds']:.2f}s")

            console.print(table)
            console.print()

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            logger.error("Indexing failed", error=str(e))
            raise typer.Exit(1)

    # Run async
    asyncio.run(run_indexing())


@app.command()
def list_corpora():
    """
    List all indexed corpora.
    """
    console.print("\n[bold blue]Available Corpora[/bold blue]\n")

    async def run_list():
        try:
            from app.db.connection import get_db_session
            from app.db.models import Corpus, Document, Chunk
            from sqlalchemy import select, func

            async with get_db_session() as session:
                # Get all corpora with counts
                result = await session.execute(
                    select(
                        Corpus,
                        func.count(Document.id.distinct()).label('doc_count'),
                        func.count(Chunk.id).label('chunk_count')
                    )
                    .outerjoin(Document, Corpus.id == Document.corpus_id)
                    .outerjoin(Chunk, Corpus.id == Chunk.corpus_id)
                    .group_by(Corpus.id)
                )

                rows = result.all()

                if not rows:
                    console.print("[yellow]No corpora found. Use 'cli.py index' to create one.[/yellow]\n")
                    return

                table = Table(title="Indexed Corpora")
                table.add_column("Name", style="cyan")
                table.add_column("Documents", style="green")
                table.add_column("Chunks", style="green")
                table.add_column("Path", style="dim")

                for corpus, doc_count, chunk_count in rows:
                    table.add_row(
                        corpus.name,
                        str(doc_count),
                        str(chunk_count),
                        corpus.file_path
                    )

                console.print(table)
                console.print()

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(run_list())


@app.command()
def check():
    """
    Check system health and configuration.
    """
    console.print("\n[bold blue]System Health Check[/bold blue]\n")

    async def run_check():
        try:
            # Check database
            with console.status("[bold green]Checking database...[/bold green]"):
                await init_db()
                db_ok = await check_db_connection()

            if db_ok:
                console.print("[green]✓[/green] Database: Connected")
            else:
                console.print("[red]✗[/red] Database: Failed")
                raise typer.Exit(1)

            # Check extensions
            extensions = await check_extensions()
            for ext, ok in extensions.items():
                if ok:
                    console.print(f"[green]✓[/green] Extension '{ext}': Installed")
                else:
                    console.print(f"[red]✗[/red] Extension '{ext}': Missing")

            # Check configuration
            console.print(f"\n[bold]Configuration:[/bold]")
            console.print(f"  Database URL: {settings.database_url}")
            console.print(f"  Corpus Path: {settings.corpus_base_path}")
            console.print(f"  LLM Model: {settings.default_model}")
            console.print(f"  Embedding Model: {settings.embedding_model}")

            console.print("\n[green]System is healthy![/green]\n")

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(run_check())


@app.command()
def info(corpus_name: str):
    """
    Get information about a specific corpus.
    """
    console.print(f"\n[bold blue]Corpus Information: {corpus_name}[/bold blue]\n")

    async def run_info():
        try:
            from app.db.connection import get_db_session
            from app.db.models import Corpus
            from app.ingestion.indexer import DocumentIndexer
            from sqlalchemy import select

            async with get_db_session() as session:
                # Get corpus
                result = await session.execute(
                    select(Corpus).where(Corpus.name == corpus_name)
                )
                corpus = result.scalar_one_or_none()

                if not corpus:
                    console.print(f"[red]Corpus not found: {corpus_name}[/red]")
                    raise typer.Exit(1)

                # Get stats
                indexer = DocumentIndexer()
                stats = await indexer.get_corpus_stats(corpus.id)

                # Display info
                table = Table(title=f"Corpus: {corpus_name}")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("ID", stats["corpus_id"])
                table.add_row("Name", stats["corpus_name"])
                table.add_row("Description", stats["description"] or "N/A")
                table.add_row("File Path", stats["file_path"])
                table.add_row("Documents", str(stats["document_count"]))
                table.add_row("Chunks", str(stats["chunk_count"]))
                table.add_row("Created", stats["created_at"])
                table.add_row("Updated", stats["updated_at"])

                console.print(table)
                console.print()

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(run_info())


if __name__ == "__main__":
    app()
