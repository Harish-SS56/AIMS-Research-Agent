"""
Logging utilities for the Agentic Deep Research System.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from typing import Any, Dict, List, Optional

# Rich console for pretty output
console = Console()

# Configure logging
def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Setup logging with rich handler."""
    handlers = [RichHandler(console=console, rich_tracebacks=True)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class AgentTrace:
    """Trace execution of agent for debugging and visualization."""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.query: str = ""
        self.final_answer: str = ""
        self.citations: List[str] = []
    
    def start(self, query: str):
        """Start tracing a new query."""
        self.query = query
        self.start_time = datetime.now()
        self.steps = []
    
    def add_step(
        self,
        step_type: str,
        input_data: Any = None,
        output_data: Any = None,
        metadata: Optional[Dict] = None
    ):
        """Add a step to the trace."""
        self.steps.append({
            "timestamp": datetime.now().isoformat(),
            "type": step_type,
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {}
        })
    
    def end(self, answer: str, citations: List[str]):
        """End tracing."""
        self.end_time = datetime.now()
        self.final_answer = answer
        self.citations = citations
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def tool_calls(self) -> int:
        """Count tool calls (retrieval operations)."""
        return len([s for s in self.steps if s["type"] in ["retrieve", "search"]])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "query": self.query,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "citations": self.citations
        }
    
    def print_summary(self):
        """Print a summary of the trace."""
        table = Table(title="Agent Execution Trace")
        table.add_column("Step", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Details", style="green")
        
        for i, step in enumerate(self.steps, 1):
            details = str(step.get("output", ""))[:100] + "..." if len(str(step.get("output", ""))) > 100 else str(step.get("output", ""))
            table.add_row(str(i), step["type"], details)
        
        console.print(table)
        console.print(f"\n[bold]Duration:[/bold] {self.duration:.2f}s")
        console.print(f"[bold]Tool Calls:[/bold] {self.tool_calls}")
        console.print(f"[bold]Citations:[/bold] {len(self.citations)}")


def print_header(text: str):
    """Print a header."""
    console.print(Panel(text, style="bold blue"))


def print_success(text: str):
    """Print success message."""
    console.print(f"[green]✓[/green] {text}")


def print_error(text: str):
    """Print error message."""
    console.print(f"[red]✗[/red] {text}")


def print_warning(text: str):
    """Print warning message."""
    console.print(f"[yellow]![/yellow] {text}")


def print_info(text: str):
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {text}")
