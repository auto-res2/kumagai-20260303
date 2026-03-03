"""
Evaluation script for comparing C3oT and baseline methods.
Fetches results from WandB and generates comparison metrics and visualizations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory containing results"
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help="JSON string list of run IDs to compare",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (defaults to WANDB_ENTITY env var)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project (defaults to WANDB_PROJECT env var)",
    )
    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> Optional[Dict]:
    """
    Fetch run data from WandB API by display name.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dict with run data or None if not found
    """
    api = wandb.Api()

    try:
        # Filter by display name (most recent)
        runs = api.runs(
            f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
        )

        if not runs:
            print(f"Warning: No runs found for {run_id}")
            return None

        run = runs[0]  # most recent

        # Extract data
        summary = dict(run.summary)
        config = dict(run.config)

        # Get history for time series metrics
        history = run.history()

        return {
            "run_id": run_id,
            "wandb_id": run.id,
            "summary": summary,
            "config": config,
            "history": history,
        }

    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return None


def export_per_run_metrics(run_data: Dict, results_dir: Path):
    """
    Export per-run metrics to JSON and create figures.

    Args:
        run_data: Data for a single run
        results_dir: Base results directory
    """
    run_id = run_data["run_id"]
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Export metrics
    metrics_file = run_dir / "wandb_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(run_data["summary"], f, indent=2)
    print(f"Exported metrics for {run_id} to {metrics_file}")

    # Create per-run visualizations if history exists
    history = run_data.get("history")
    if history is not None and not history.empty:
        # Sample visualization: plot any numeric metrics over time/step
        numeric_cols = history.select_dtypes(include=["number"]).columns

        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            for col in numeric_cols:
                if col != "_step":
                    ax.plot(history.index, history[col], label=col, marker="o")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.set_title(f"Metrics over time: {run_id}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            fig_file = run_dir / f"{run_id}_metrics.pdf"
            plt.savefig(fig_file, bbox_inches="tight")
            plt.close()
            print(f"Saved figure to {fig_file}")


def aggregate_and_compare(run_data_list: List[Dict], results_dir: Path):
    """
    Aggregate metrics across runs and create comparison visualizations.

    Args:
        run_data_list: List of run data dicts
        results_dir: Base results directory
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics from all runs
    all_metrics = {}
    for run_data in run_data_list:
        run_id = run_data["run_id"]
        summary = run_data["summary"]
        all_metrics[run_id] = summary

    # Determine primary metric
    primary_metric = "reliability_per_token"

    # Find best proposed and best baseline
    proposed_runs = {
        rid: metrics
        for rid, metrics in all_metrics.items()
        if "proposed" in rid.lower()
    }
    baseline_runs = {
        rid: metrics
        for rid, metrics in all_metrics.items()
        if "comparative" in rid.lower() or "baseline" in rid.lower()
    }

    best_proposed = None
    best_baseline = None

    if proposed_runs:
        best_proposed = max(
            proposed_runs.items(), key=lambda x: x[1].get(primary_metric, 0.0)
        )

    if baseline_runs:
        best_baseline = max(
            baseline_runs.items(), key=lambda x: x[1].get(primary_metric, 0.0)
        )

    # Calculate gap
    gap = None
    if best_proposed and best_baseline:
        proposed_val = best_proposed[1].get(primary_metric, 0.0)
        baseline_val = best_baseline[1].get(primary_metric, 0.0)
        gap = proposed_val - baseline_val

    # Export aggregated metrics
    aggregated = {
        "primary_metric": primary_metric,
        "metrics_by_run": all_metrics,
        "best_proposed": {
            "run_id": best_proposed[0],
            "value": best_proposed[1].get(primary_metric, 0.0),
        }
        if best_proposed
        else None,
        "best_baseline": {
            "run_id": best_baseline[0],
            "value": best_baseline[1].get(primary_metric, 0.0),
        }
        if best_baseline
        else None,
        "gap": gap,
    }

    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nExported aggregated metrics to {agg_file}")

    # Create comparison visualizations
    create_comparison_plots(all_metrics, comparison_dir)


def create_comparison_plots(all_metrics: Dict[str, Dict], comparison_dir: Path):
    """
    Create comparison plots for all runs.

    Args:
        all_metrics: Dict mapping run_id to metrics
        comparison_dir: Directory to save plots
    """
    # Key metrics to compare
    metrics_to_plot = [
        "final_answer_accuracy",
        "reliability_per_token",
        "avg_tokens_per_sample",
        "cost_usd",
    ]

    # Prepare data for plotting
    plot_data = []
    for run_id, metrics in all_metrics.items():
        for metric_name in metrics_to_plot:
            value = metrics.get(metric_name)
            if value is not None:
                plot_data.append(
                    {"run_id": run_id, "metric": metric_name, "value": value}
                )

    if not plot_data:
        print("Warning: No metrics to plot")
        return

    df = pd.DataFrame(plot_data)

    # Create individual plots for each metric
    for metric_name in metrics_to_plot:
        metric_df = df[df["metric"] == metric_name]
        if metric_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar plot
        sns.barplot(data=metric_df, x="run_id", y="value", ax=ax, palette="viridis")

        ax.set_xlabel("Run ID", fontsize=12)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        ax.set_title(
            f"Comparison: {metric_name.replace('_', ' ').title()}", fontsize=14
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        fig_file = comparison_dir / f"comparison_{metric_name}.pdf"
        plt.savefig(fig_file, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison plot to {fig_file}")

    # Create overall comparison plot (all metrics normalized)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Normalize each metric to [0, 1] for comparison
    normalized_data = []
    for metric_name in metrics_to_plot:
        metric_df = df[df["metric"] == metric_name]
        if not metric_df.empty:
            max_val = metric_df["value"].max()
            min_val = metric_df["value"].min()
            if max_val > min_val:
                for _, row in metric_df.iterrows():
                    normalized_data.append(
                        {
                            "run_id": row["run_id"],
                            "metric": metric_name,
                            "normalized_value": (row["value"] - min_val)
                            / (max_val - min_val),
                        }
                    )

    if normalized_data:
        norm_df = pd.DataFrame(normalized_data)
        pivot_df = norm_df.pivot(
            index="run_id", columns="metric", values="normalized_value"
        )

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            ax=ax,
            cbar_kws={"label": "Normalized Value"},
        )

        ax.set_title("Normalized Metric Comparison (Higher is Better)", fontsize=14)
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Run ID", fontsize=12)

        plt.tight_layout()

        fig_file = comparison_dir / "comparison_heatmap.pdf"
        plt.savefig(fig_file, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmap to {fig_file}")


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")

    # Get WandB credentials
    entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    project = args.wandb_project or os.getenv("WANDB_PROJECT", "kumagai")

    if not entity:
        raise ValueError(
            "WandB entity not specified (use --wandb_entity or WANDB_ENTITY env var)"
        )

    print(f"WandB: {entity}/{project}")

    # Results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data for each run
    print("\nFetching run data from WandB...")
    run_data_list = []
    for run_id in run_ids:
        print(f"  Fetching {run_id}...")
        run_data = fetch_run_data(entity, project, run_id)
        if run_data:
            run_data_list.append(run_data)
            # Export per-run metrics
            export_per_run_metrics(run_data, results_dir)

    if not run_data_list:
        print("Error: No runs found")
        return

    # Aggregate and compare
    print("\nGenerating comparison metrics and plots...")
    aggregate_and_compare(run_data_list, results_dir)

    print("\n=== Evaluation Complete ===")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
