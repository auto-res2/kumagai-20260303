"""
Data preprocessing module for GSM8K dataset.
Handles loading, filtering, and preparation of math word problems.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
import random

from datasets import load_dataset


def extract_numeric_answer(answer_text: str) -> Optional[float]:
    """
    Extract the final numeric answer from GSM8K answer format.
    GSM8K answers end with "#### NUMBER"

    Args:
        answer_text: Full answer text from GSM8K

    Returns:
        Numeric answer or None if not found
    """
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer_text)
    if match:
        # Remove commas and convert to float
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None
    return None


def load_gsm8k_subset(
    split: str = "test",
    num_samples: int = 200,
    seed: int = 42,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Load a deterministic subset of GSM8K dataset.

    Args:
        split: Dataset split (train/test)
        num_samples: Number of samples to load
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle before sampling
        cache_dir: Cache directory for HuggingFace datasets

    Returns:
        List of dictionaries with keys: question, answer, numeric_answer, idx
    """
    # Load full dataset
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    # Convert to list for easier manipulation
    all_examples = []
    for idx, example in enumerate(dataset):
        question = example["question"]
        answer_text = example["answer"]
        numeric_answer = extract_numeric_answer(answer_text)

        # Only include examples with valid numeric answers
        if numeric_answer is not None:
            all_examples.append(
                {
                    "idx": idx,
                    "question": question,
                    "answer_text": answer_text,
                    "numeric_answer": numeric_answer,
                }
            )

    # Deterministic sampling
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_examples)

    # Take requested number of samples
    subset = all_examples[:num_samples]

    print(f"Loaded {len(subset)} examples from GSM8K {split} split")

    return subset


def prepare_example_for_inference(example: Dict) -> Dict:
    """
    Prepare a single example for inference.

    Args:
        example: Raw example from dataset

    Returns:
        Prepared example with standardized format
    """
    return {
        "idx": example["idx"],
        "question": example["question"].strip(),
        "ground_truth": example["numeric_answer"],
        "answer_text": example["answer_text"],
    }


def get_dataset(cfg) -> List[Dict]:
    """
    Main entry point for loading dataset based on Hydra config.

    Args:
        cfg: Hydra config object with dataset parameters

    Returns:
        List of prepared examples
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Accessing cfg.dataset when it's actually cfg.run.dataset
    # [CAUSE]: Hydra loads run configs as nested under cfg.run
    # [FIX]: Access cfg.run.dataset instead of cfg.dataset
    #
    # [OLD CODE]:
    # raw_examples = load_gsm8k_subset(
    #     split=cfg.dataset.split,
    #     num_samples=cfg.dataset.num_samples,
    #     seed=cfg.dataset.seed,
    #     shuffle=cfg.dataset.shuffle,
    #     cache_dir=cfg.get("cache_dir", ".cache/"),
    #
    # [NEW CODE]:
    # Load subset
    raw_examples = load_gsm8k_subset(
        split=cfg.run.dataset.split,
        num_samples=cfg.run.dataset.num_samples,
        seed=cfg.run.dataset.seed,
        shuffle=cfg.run.dataset.shuffle,
        cache_dir=cfg.get("cache_dir", ".cache/"),
    )

    # Prepare for inference
    prepared_examples = [prepare_example_for_inference(ex) for ex in raw_examples]

    return prepared_examples


if __name__ == "__main__":
    # Quick test
    examples = load_gsm8k_subset(num_samples=5, seed=42)
    for ex in examples[:3]:
        print(f"\nQuestion: {ex['question'][:100]}...")
        print(f"Answer: {ex['numeric_answer']}")
