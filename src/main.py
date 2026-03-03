"""
Main orchestrator for C3oT experiment.
Handles inference runs with Hydra configuration and WandB logging.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.preprocess import get_dataset
from src.model import create_llm
from src.inference import run_inference


def apply_mode_overrides(cfg: DictConfig, mode: str) -> DictConfig:
    """
    Apply mode-specific overrides to config.

    Args:
        cfg: Original config
        mode: Execution mode (main, sanity_check, pilot)

    Returns:
        Modified config
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: ConfigAttributeError: Key 'dataset' is not in struct
    # [CAUSE]: The dataset config is nested under cfg.run.dataset, not cfg.dataset. Hydra loads the run config as a nested structure.
    # [FIX]: Access cfg.run.dataset instead of cfg.dataset, and set struct=False to allow modifications
    #
    # [OLD CODE]:
    # if mode == "sanity_check":
    #     cfg.dataset.num_samples = 10
    #     cfg.wandb.project = f"{cfg.wandb.project}-sanity"
    #     print(f"Sanity check mode: reduced to {cfg.dataset.num_samples} samples")
    #
    # [NEW CODE]:
    if mode == "sanity_check":
        # Override for quick validation
        OmegaConf.set_struct(cfg, False)

        # Access dataset config under cfg.run (where Hydra loads it)
        cfg.run.dataset.num_samples = 10
        cfg.wandb.project = f"{cfg.wandb.project}-sanity"
        print(f"Sanity check mode: reduced to {cfg.run.dataset.num_samples} samples")

        OmegaConf.set_struct(cfg, True)

    return cfg


def calculate_metrics(results: List[Dict], cfg: DictConfig) -> Dict:
    """
    Calculate aggregate metrics from results.

    Args:
        results: List of inference results
        cfg: Hydra config

    Returns:
        Dict of metrics
    """
    if not results:
        return {}

    # Basic accuracy
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    # Token usage
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    total_cost = sum(r.get("total_cost_usd", 0.0) for r in results)
    avg_tokens = total_tokens / total if total > 0 else 0.0

    # C3oT specific metrics
    if cfg.run.method.name == "c3ot":
        commitment_valid = sum(1 for r in results if r.get("commitment_valid", False))
        invariants_satisfied = sum(
            1 for r in results if r.get("invariants_satisfied", False)
        )
        repaired = sum(1 for r in results if r.get("repaired", False))

        commitment_rate = commitment_valid / total if total > 0 else 0.0
        invariant_rate = invariants_satisfied / total if total > 0 else 0.0
        repair_rate = repaired / total if total > 0 else 0.0

        # Reliability per token (proposed metric)
        # Higher is better: correct answers with fewer tokens
        reliability_per_token = (
            (accuracy * 1000) / avg_tokens if avg_tokens > 0 else 0.0
        )

        return {
            "final_answer_accuracy": accuracy,
            "commitment_consistency": commitment_rate,
            "invariant_satisfaction": invariant_rate,
            "repair_rate": repair_rate,
            "tokens_used": total_tokens,
            "avg_tokens_per_sample": avg_tokens,
            "cost_usd": total_cost,
            "reliability_per_token": reliability_per_token,
            "total_samples": total,
            "correct_count": correct,
        }

    else:  # baseline
        # Reliability per token for baseline
        reliability_per_token = (
            (accuracy * 1000) / avg_tokens if avg_tokens > 0 else 0.0
        )

        return {
            "final_answer_accuracy": accuracy,
            "commitment_consistency": None,
            "constraint_coverage": None,
            "invariant_satisfaction": None,
            "repair_rate": 0.0,
            "tokens_used": total_tokens,
            "avg_tokens_per_sample": avg_tokens,
            "cost_usd": total_cost,
            "reliability_per_token": reliability_per_token,
            "total_samples": total,
            "correct_count": correct,
        }


def validate_sanity(results: List[Dict], metrics: Dict, cfg: DictConfig, mode: str):
    """
    Perform sanity validation and emit verdict.

    Args:
        results: List of inference results
        metrics: Calculated metrics
        cfg: Hydra config
        mode: Execution mode
    """
    if mode != "sanity_check":
        return

    # Check basic requirements
    total = len(results)

    # Inference tasks require at least 5 samples processed
    if total < 5:
        print(
            f"SANITY_VALIDATION: FAIL reason=insufficient_samples (got {total}, need >= 5)"
        )
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'status': 'fail'})}"
        )
        return

    # Check that we have valid outputs
    valid_outputs = sum(1 for r in results if r.get("final_answer") is not None)

    if valid_outputs < 5:
        print(
            f"SANITY_VALIDATION: FAIL reason=insufficient_valid_outputs (got {valid_outputs})"
        )
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'valid_outputs': valid_outputs, 'status': 'fail'})}"
        )
        return

    # Check that all metrics are finite
    if not all(
        isinstance(v, (int, float, type(None)))
        and (
            v is None
            or (
                not isinstance(v, float)
                or (v == v and v != float("inf") and v != float("-inf"))
            )
        )
        for v in metrics.values()
    ):
        print(f"SANITY_VALIDATION: FAIL reason=non_finite_metrics")
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'status': 'fail'})}"
        )
        return

    # Check that not all outputs are identical (would indicate broken inference)
    unique_answers = len(
        set(r.get("final_answer") for r in results if r.get("final_answer") is not None)
    )

    if unique_answers < 2:
        print(f"SANITY_VALIDATION: FAIL reason=all_identical_outputs")
        print(
            f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': total, 'unique_answers': unique_answers, 'status': 'fail'})}"
        )
        return

    # Success
    print("SANITY_VALIDATION: PASS")
    summary = {
        "samples": total,
        "valid_outputs": valid_outputs,
        "unique_answers": unique_answers,
        "accuracy": metrics.get("final_answer_accuracy", 0.0),
        "status": "pass",
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for inference execution.

    Args:
        cfg: Hydra configuration
    """
    # Apply mode overrides
    mode = cfg.get("mode", "main")
    cfg = apply_mode_overrides(cfg, mode)

    print(f"=== C3oT Experiment ===")
    print(f"Mode: {mode}")
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Dataset: {cfg.run.dataset.name} ({cfg.run.dataset.num_samples} samples)")
    print()

    # Initialize WandB
    wandb_enabled = cfg.wandb.mode == "online"

    if wandb_enabled:
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB run: {wandb.run.url}")
    else:
        print("WandB disabled")

    # Load dataset
    print("\nLoading dataset...")
    examples = get_dataset(cfg)
    print(f"Loaded {len(examples)} examples")

    # Create LLM client
    print(f"\nInitializing LLM: {cfg.llm.provider}/{cfg.llm.model}")
    llm = create_llm(cfg)

    # Run inference
    print(f"\nRunning inference with {cfg.run.method.name}...")
    results = run_inference(examples, llm, cfg, mode)
    print(f"Completed {len(results)} inferences")

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results, cfg)

    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value}")

    # Log to WandB
    if wandb_enabled:
        wandb.log(metrics)
        for key, value in metrics.items():
            if value is not None:
                wandb.summary[key] = value

    # Save results to disk
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Get LLM stats
    llm_stats = llm.get_stats()
    print(f"\nTotal LLM usage:")
    print(f"  Tokens: {llm_stats['total_tokens']}")
    print(f"  Cost: ${llm_stats['total_cost_usd']:.4f}")

    # Sanity validation
    validate_sanity(results, metrics, cfg, mode)

    # Finish WandB
    if wandb_enabled:
        wandb.finish()

    print("\n=== Experiment Complete ===")


if __name__ == "__main__":
    main()
