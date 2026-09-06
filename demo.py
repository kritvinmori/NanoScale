import time
from nanoscale.engine import NanoScaleEngine
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_demo():
    console.rule("[bold cyan]NanoScale vs. Stock 1.5B Head-to-Head[/bold cyan]")
    engine = NanoScaleEngine()

    test_prompt = "An electronics store offers a 20% discount on a $400 laptop. A member receives an additional 10% discount off the already discounted price. What is the final price before tax?"

    console.print(f"\n[bold yellow]Target Problem:[/bold yellow] {test_prompt}\n")

    # 1. Stock Baseline
    console.print("[dim]--> Running Stock Qwen-1.5B (Greedy Single-shot)...[/dim]")
    t0 = time.time()
    stock_output = engine.run_baseline(test_prompt, max_tokens=350)
    stock_time = time.time() - t0

    # 2. NanoScale Test-Time Compute
    console.print("[dim]--> Running NanoScale (3-Path Test-Time Consensus)...[/dim]")
    t1 = time.time()
    nanoscale_res = engine.generate(test_prompt, n_candidates=3, max_new_tokens=350)
    nano_time = time.time() - t1

    # Candidate Verification Table
    table = Table(title="Parallel Trajectory Verification & Selection", show_lines=True)
    table.add_column("Path", width=6, justify="center")
    table.add_column("PRM Score", width=12, justify="center")
    table.add_column("Answer", width=12, justify="center", style="bold yellow")
    table.add_column("Ending Snippet", width=68)

    winning_idx = nanoscale_res["all_candidates"].index(nanoscale_res["selected_solution"])
    for i, (text, score, ans) in enumerate(zip(nanoscale_res["all_candidates"], nanoscale_res["all_scores"], nanoscale_res["all_answers"])):
        winner = (i == winning_idx)
        score_str = f"[bold green]{score:.4f} (Win)[/bold green]" if winner else f"[dim]{score:.4f}[/dim]"
        ans_str = f"[bold green]{ans}[/bold green]" if winner else f"[dim]{ans}[/dim]"
        snippet = "..." + text[-110:].replace("\n", " ")
        table.add_row(f"Path {i+1}", score_str, ans_str, snippet)

    console.print(table)

    # Comparison Panels
    console.print(Panel(
        stock_output,
        title=f"[bold red]Stock Qwen2.5-1.5B Baseline (Latency: {stock_time:.2f}s)[/bold red]",
        border_style="red"
    ))

    consensus_label = "[Consensus]" if nanoscale_res["has_consensus"] else "[PRM Top-Score]"
    console.print(Panel(
        nanoscale_res["selected_solution"],
        title=f"[bold green]NanoScale Output | Ans: {nanoscale_res['extracted_answer']} {consensus_label} (Latency: {nano_time:.2f}s)[/bold green]",
        border_style="green"
    ))

if __name__ == "__main__":
    run_demo()