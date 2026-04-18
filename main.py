"""
Hybrid Edge-Cloud Document Agent
=================================
Main entry point - CLI + interactive demo runner.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.agent import HybridDocumentAgent
from core.config import AgentConfig
from core.models import Query, Document

console = Console()


async def interactive_demo(agent: HybridDocumentAgent):
    """Run interactive CLI demo."""
    console.print(Panel.fit(
        "[bold cyan]Hybrid Edge-Cloud Document Agent[/bold cyan]\n"
        "[dim]Intelligent Query Routing · Cost-Optimized RAG[/dim]",
        border_style="cyan"
    ))

    # Load sample documents
    sample_docs = [
        Document(
            id="doc_001",
            title="Employee Handbook v3.2",
            content="""
            Section 4.3 - Remote Work Policy:
            Employees may work remotely up to 4 days per week with manager approval.
            Home office equipment allowance is $800 per year.
            All remote workers must maintain core hours of 10am-3pm in their local timezone.
            VPN usage is mandatory for all remote work sessions.
            Personal devices must be enrolled in the MDM system before accessing company resources.

            Section 5.1 - Leave Policy:
            Annual leave: 20 days per year. Sick leave: 10 days per year.
            Parental leave: 16 weeks fully paid for primary caregivers, 4 weeks for secondary.
            Leave requests must be submitted 2 weeks in advance via the HR portal.

            Section 6.2 - Code of Conduct:
            All employees must adhere to the company's ethical guidelines.
            Conflicts of interest must be disclosed to HR within 30 days.
            """,
            metadata={"type": "policy", "department": "HR", "version": "3.2"}
        ),
        Document(
            id="doc_002",
            title="IT Security Guidelines 2024",
            content="""
            Section 2.1 - Remote Access Security:
            Remote workers must use approved VPN at all times when accessing company systems.
            Multi-factor authentication (MFA) is mandatory for all remote logins.
            Data classification level 3+ requires corporate-issued hardware only.
            Personal devices are prohibited from storing confidential company data.

            Section 3.4 - Password Policy:
            Passwords must be minimum 12 characters with complexity requirements.
            Password rotation every 90 days for privileged accounts.
            Password managers are approved and recommended for all staff.

            Section 5.0 - Incident Response:
            Security incidents must be reported within 1 hour of discovery.
            Contact security@company.com or call the 24/7 hotline: ext 9911.
            """,
            metadata={"type": "security", "department": "IT", "classification": "internal"}
        ),
        Document(
            id="doc_003",
            title="Q4 2024 Financial Report",
            content="""
            Executive Summary:
            Q4 2024 revenue reached $18.7M, representing a 31% increase year-over-year.
            Full year 2024 revenue: $67.2M (+30% vs 2023's $51.8M).
            EBITDA margin improved to 23% from 19% in 2023.

            Revenue by Segment:
            - Enterprise: $41.3M (61% of total, +38% YoY)
            - SMB: $18.4M (27% of total, +15% YoY)
            - Consumer: $7.5M (11% of total, +8% YoY)

            Cost Structure:
            Cloud infrastructure costs increased 18% in H2 2024.
            Headcount grew from 312 to 387 employees (+24%).
            R&D investment: $12.1M (18% of revenue).

            Outlook 2025:
            Projected revenue: $85-92M based on current pipeline.
            Key risks: market saturation in SMB, rising infrastructure costs.
            """,
            metadata={"type": "financial", "period": "Q4-2024", "classification": "confidential"}
        ),
        Document(
            id="doc_004",
            title="Service Agreement - Standard Terms v2.1",
            content="""
            Section 8.2 - Limitation of Liability:
            Company's total liability shall not exceed the greater of $500 or fees paid
            in the 12 months preceding the claim. Consequential, indirect, incidental,
            special or punitive damages are explicitly excluded regardless of cause.

            Section 9.1 - Refund Policy:
            Digital products may be refunded within 30 days if defective or not as described.
            Service subscriptions may be cancelled with 30 days written notice.
            Refund requests must be submitted via support@company.com with order ID.
            Pro-rated refunds available for annual plans cancelled after 3 months.

            Section 11.4 - Force Majeure:
            Neither party shall be liable for failure to perform obligations due to causes
            beyond reasonable control including acts of God, government actions, natural
            disasters, pandemics, or infrastructure failures outside the party's control.

            Section 14.0 - SLA Terms:
            Critical incidents (P1): Response within 1 hour, resolution target 4 hours.
            High priority (P2): Response within 4 hours, resolution target 24 hours.
            Standard (P3): Response within 1 business day, resolution target 5 days.
            """,
            metadata={"type": "legal", "version": "2.1", "classification": "internal"}
        ),
    ]

    console.print(f"\n[green]✓[/green] Loaded [bold]{len(sample_docs)}[/bold] documents into corpus\n")

    # Index documents
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Indexing documents...", total=None)
        await agent.index_documents(sample_docs)
        progress.update(task, description="[green]Documents indexed successfully[/green]")

    # Demo queries
    demo_queries = [
        "What is the refund policy for digital products?",
        "Find all clauses related to liability limitations in the service agreement.",
        "What are the SLA response times for critical incidents?",
        "Analyze the revenue trends across all financial reports and identify key growth drivers.",
        "Cross-reference the remote work policy with IT security requirements for remote employees.",
        "What is force majeure and how does it apply in our contracts?",
        "Compare enterprise vs SMB revenue performance and forecast implications.",
    ]

    console.print("\n[bold cyan]── Running Demo Queries ──[/bold cyan]\n")

    results_summary = []

    for i, q_text in enumerate(demo_queries, 1):
        console.print(f"[dim]Query {i}/{len(demo_queries)}:[/dim] [yellow]{q_text}[/yellow]")

        query = Query(text=q_text, session_id="demo_session")
        result = await agent.process(query)

        route_color = {"edge": "cyan", "cloud": "magenta", "hybrid": "green", "cache": "yellow"}.get(result.route, "white")
        console.print(
            f"  [bold {route_color}]→ {result.route.upper()}[/bold {route_color}] "
            f"[dim]({result.model_used})[/dim] | "
            f"⏱ {result.latency_ms}ms | "
            f"💰 ${result.cost:.5f} | "
            f"📊 complexity={result.complexity_score:.2f}"
        )
        console.print(f"  [dim]{result.response[:120]}...[/dim]\n")

        results_summary.append(result)

    # Print session stats
    stats = agent.get_session_stats()
    print_stats_table(stats, results_summary)


def print_stats_table(stats: dict, results: list):
    """Print a summary stats table."""
    console.print("\n[bold cyan]── Session Statistics ──[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Total Queries", str(stats["total_queries"]))
    table.add_row("Edge Routed", f"[cyan]{stats['edge_count']}[/cyan]")
    table.add_row("Cloud Routed", f"[magenta]{stats['cloud_count']}[/magenta]")
    table.add_row("Hybrid Routed", f"[green]{stats['hybrid_count']}[/green]")
    table.add_row("Cache Hits", f"[yellow]{stats['cache_hits']}[/yellow]")
    table.add_row("Total Cost (Actual)", f"[green]${stats['total_cost_actual']:.5f}[/green]")
    table.add_row("Total Cost (Cloud-only)", f"[red]${stats['total_cost_cloud_only']:.5f}[/red]")
    table.add_row("Cost Savings", f"[bold green]${stats['cost_saved']:.5f} ({stats['savings_pct']:.1f}%)[/bold green]")
    table.add_row("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")

    console.print(table)


async def single_query_mode(agent: HybridDocumentAgent, query_text: str):
    """Process a single query and print result as JSON."""
    # Load sample documents (same as demo)
    sample_docs = [
        Document(
            id="doc_001",
            title="Employee Handbook v3.2",
            content="""
            Section 4.3 - Remote Work Policy:
            Employees may work remotely up to 4 days per week with manager approval.
            Home office equipment allowance is $800 per year.
            All remote workers must maintain core hours of 10am-3pm in their local timezone.
            VPN usage is mandatory for all remote work sessions.
            Personal devices must be enrolled in the MDM system before accessing company resources.

            Section 5.1 - Leave Policy:
            Annual leave: 20 days per year. Sick leave: 10 days per year.
            Parental leave: 16 weeks fully paid for primary caregivers, 4 weeks for secondary.
            Leave requests must be submitted 2 weeks in advance via the HR portal.

            Section 6.2 - Code of Conduct:
            All employees must adhere to the company's ethical guidelines.
            Conflicts of interest must be disclosed to HR within 30 days.
            """,
            metadata={"type": "policy", "department": "HR", "version": "3.2"}
        ),
        Document(
            id="doc_002",
            title="IT Security Guidelines 2024",
            content="""
            Section 2.1 - Remote Access Security:
            Remote workers must use approved VPN at all times when accessing company systems.
            Multi-factor authentication (MFA) is mandatory for all remote logins.
            Data classification level 3+ requires corporate-issued hardware only.
            Personal devices are prohibited from storing confidential company data.

            Section 3.4 - Password Policy:
            Passwords must be minimum 12 characters with complexity requirements.
            Password rotation every 90 days for privileged accounts.
            Password managers are approved and recommended for all staff.

            Section 5.0 - Incident Response:
            Security incidents must be reported within 1 hour of discovery.
            Contact security@company.com or call the 24/7 hotline: ext 9911.
            """,
            metadata={"type": "security", "department": "IT", "classification": "internal"}
        ),
        Document(
            id="doc_003",
            title="Q4 2024 Financial Report",
            content="""
            Executive Summary:
            Q4 2024 revenue reached $18.7M, representing a 31% increase year-over-year.
            Full year 2024 revenue: $67.2M (+30% vs 2023's $51.8M).
            EBITDA margin improved to 23% from 19% in 2023.

            Revenue by Segment:
            - Enterprise: $41.3M (61% of total, +38% YoY)
            - SMB: $18.4M (27% of total, +15% YoY)
            - Consumer: $7.5M (11% of total, +8% YoY)

            Cost Structure:
            Cloud infrastructure costs increased 18% in H2 2024.
            Headcount grew from 312 to 387 employees (+24%).
            R&D investment: $12.1M (18% of revenue).

            Outlook 2025:
            Projected revenue: $85-92M based on current pipeline.
            Key risks: market saturation in SMB, rising infrastructure costs.
            """,
            metadata={"type": "financial", "period": "Q4-2024", "classification": "confidential"}
        ),
        Document(
            id="doc_004",
            title="Service Agreement - Standard Terms v2.1",
            content="""
            Section 8.2 - Limitation of Liability:
            Company's total liability shall not exceed the greater of $500 or fees paid
            in the 12 months preceding the claim. Consequential, indirect, incidental,
            special or punitive damages are explicitly excluded regardless of cause.

            Section 9.1 - Refund Policy:
            Digital products may be refunded within 30 days if defective or not as described.
            Service subscriptions may be cancelled with 30 days written notice.
            Refund requests must be submitted via support@company.com with order ID.
            Pro-rated refunds available for annual plans cancelled after 3 months.

            Section 11.4 - Force Majeure:
            Neither party shall be liable for failure to perform obligations due to causes
            beyond reasonable control including acts of God, government actions, natural
            disasters, pandemics, or infrastructure failures outside the party's control.

            Section 14.0 - SLA Terms:
            Critical incidents (P1): Response within 1 hour, resolution target 4 hours.
            High priority (P2): Response within 4 hours, resolution target 24 hours.
            Standard (P3): Response within 1 business day, resolution target 5 days.
            """,
            metadata={"type": "legal", "version": "2.1", "classification": "internal"}
        ),
    ]

    # Index documents
    await agent.index_documents(sample_docs)

    # Process query
    query = Query(text=query_text, session_id="cli_session")
    result = await agent.process(query)
    print(json.dumps(result.to_dict(), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Hybrid Edge-Cloud Document Agent")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--config", "-c", type=str, help="Path to config YAML file")
    parser.add_argument("--edge-only", action="store_true", help="Force edge routing")
    parser.add_argument("--cloud-only", action="store_true", help="Force cloud routing")
    args = parser.parse_args()

    # Build config
    config = AgentConfig()
    if args.config:
        config = AgentConfig.from_yaml(args.config)
    if args.edge_only:
        config.routing.force_route = "edge"
    if args.cloud_only:
        config.routing.force_route = "cloud"

    # Build agent
    agent = HybridDocumentAgent(config=config)

    # Run
    if args.query:
        asyncio.run(single_query_mode(agent, args.query))
    else:
        asyncio.run(interactive_demo(agent))


if __name__ == "__main__":
    main()
