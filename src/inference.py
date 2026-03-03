"""
Inference logic for C3oT (Commitment-Checked Chain-of-Thought) and baseline CoT.
Implements the two-pass commitment mechanism with external checking and repair.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.model import LLMInference


# Prompts for C3oT method
C3OT_COMMIT_PROMPT = """You are solving a grade-school math word problem using a structured approach.

TASK: Read the problem and output a JSON commitment that declares:
1. Variables and their units (if applicable, can be null)
2. Key constraints extracted from the problem
3. A plan with 2-4 steps
4. 1-3 checkable invariants (equations, inequalities, conservation laws, etc.) that MUST hold for the final answer

Do NOT solve yet. Only commit to a structure.

Problem: {question}

Output your commitment as valid JSON with this schema:
{{
  "variables": {{"var_name": "unit or null"}},
  "constraints": ["constraint1", "constraint2"],
  "plan": ["step1", "step2", "step3"],
  "invariants": ["invariant1 (must be checkable)", "invariant2"]
}}

JSON:"""

C3OT_SOLVE_PROMPT = """Now solve the problem using your commitments.

Problem: {question}

Your commitments:
{commitments}

Provide a concise solution and output:
1. Your step-by-step reasoning (brief)
2. FINAL: <number> (the numeric answer)
3. Evidence table showing how each invariant is satisfied

Format:
Reasoning: ...
FINAL: <number>
Evidence: invariant1 -> <substitution>; invariant2 -> <substitution>"""

C3OT_REPAIR_PROMPT = """Your previous attempt failed validation:
{error_message}

Problem: {question}
Previous commitments: {commitments}
Previous answer: FINAL: {previous_answer}

You must explicitly change EITHER:
- The invariants/constraints in your commitment, OR
- The final answer

Provide a corrected solution with:
FINAL: <number>
Explanation: <why you changed what you changed>"""

# Prompt for baseline CoT
BASELINE_COT_PROMPT = """Solve this grade-school math word problem step-by-step.

Problem: {question}

Provide your reasoning and then output the final numeric answer in this format:
FINAL: <number>

Show your work:"""


class CommitmentChecker:
    """External checker for C3oT commitments and invariants."""

    @staticmethod
    def validate_json(text: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Validate JSON structure of commitment.

        Returns:
            (is_valid, parsed_json, error_message)
        """
        try:
            # Extract JSON from text (in case there's extra text)
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                return False, None, "No JSON found in response"

            parsed = json.loads(json_match.group(0))

            # Check required fields
            required = ["variables", "constraints", "plan", "invariants"]
            for field in required:
                if field not in parsed:
                    return False, None, f"Missing required field: {field}"

            # Check types
            if not isinstance(parsed["variables"], dict):
                return False, None, "variables must be a dict"
            if not isinstance(parsed["constraints"], list):
                return False, None, "constraints must be a list"
            if not isinstance(parsed["plan"], list):
                return False, None, "plan must be a list"
            if not isinstance(parsed["invariants"], list):
                return False, None, "invariants must be a list"

            # Check counts
            if not (2 <= len(parsed["plan"]) <= 4):
                return False, None, "plan must have 2-4 steps"
            if not (1 <= len(parsed["invariants"]) <= 3):
                return False, None, "invariants must have 1-3 items"

            return True, parsed, ""

        except json.JSONDecodeError as e:
            return False, None, f"JSON parse error: {e}"

    @staticmethod
    def extract_final_answer(text: str) -> Optional[float]:
        """Extract FINAL: <number> from response."""
        match = re.search(r"FINAL:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                return None
        return None

    @staticmethod
    def check_invariant_satisfaction(
        commitments: Dict, final_answer: float, evidence_text: str
    ) -> Tuple[bool, str]:
        """
        Check if invariants are satisfied (heuristic check).

        This is a simplified checker that looks for:
        - Evidence lines mentioning each invariant
        - Numeric consistency (no contradictions)

        A full implementation would parse and evaluate expressions.
        """
        invariants = commitments.get("invariants", [])

        # Basic check: ensure evidence mentions each invariant
        if not evidence_text or "Evidence:" not in evidence_text:
            return False, "No evidence provided for invariants"

        evidence_lower = evidence_text.lower()

        # Check that at least one invariant is mentioned
        # (simplified: a real checker would parse and evaluate)
        has_evidence = any(inv.lower()[:20] in evidence_lower for inv in invariants)

        if not has_evidence:
            return False, "Evidence does not reference any invariants"

        return True, ""


def run_c3ot_inference(
    example: Dict, llm: LLMInference, cfg, mode: str = "main"
) -> Dict:
    """
    Run C3oT (Commitment-Checked Chain-of-Thought) inference on one example.

    Args:
        example: Dict with 'question' and 'ground_truth'
        llm: LLM inference client
        cfg: Hydra config
        mode: Execution mode (main, sanity_check, pilot)

    Returns:
        Dict with results including answer, tokens, cost, and validation status
    """
    checker = CommitmentChecker()
    question = example["question"]
    ground_truth = example["ground_truth"]

    results = {
        "idx": example["idx"],
        "question": question,
        "ground_truth": ground_truth,
        "method": "c3ot",
        "passes": [],
    }

    # Pass 1: Commitment (C-pass)
    commit_prompt = C3OT_COMMIT_PROMPT.format(question=question)
    commit_response = llm(commit_prompt)

    results["passes"].append(
        {
            "pass_type": "commitment",
            "response": commit_response["response"],
            "tokens": commit_response["tokens"],
            "cost_usd": commit_response["cost_usd"],
        }
    )

    # Validate JSON
    is_valid, commitments, error = checker.validate_json(commit_response["response"])
    results["commitment_valid"] = is_valid
    results["commitments"] = commitments

    if not is_valid:
        results["validation_error"] = error
        results["final_answer"] = None
        results["correct"] = False
        return results

    # Pass 2: Solve (S-pass)
    solve_prompt = C3OT_SOLVE_PROMPT.format(
        question=question, commitments=json.dumps(commitments, indent=2)
    )
    solve_response = llm(solve_prompt)

    results["passes"].append(
        {
            "pass_type": "solve",
            "response": solve_response["response"],
            "tokens": solve_response["tokens"],
            "cost_usd": solve_response["cost_usd"],
        }
    )

    # Extract final answer
    final_answer = checker.extract_final_answer(solve_response["response"])
    results["final_answer"] = final_answer

    if final_answer is None:
        results["validation_error"] = "No FINAL answer found"
        results["correct"] = False
        return results

    # Check invariants
    if commitments is not None:
        inv_valid, inv_error = checker.check_invariant_satisfaction(
            commitments, final_answer, solve_response["response"]
        )
        results["invariants_satisfied"] = inv_valid
    else:
        inv_valid = False
        inv_error = "No valid commitments"
        results["invariants_satisfied"] = False

    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Accessing cfg.method when it's actually cfg.run.method
    # [CAUSE]: Hydra loads run configs as nested under cfg.run
    # [FIX]: Access cfg.run.method instead of cfg.method
    #
    # [OLD CODE]:
    # if not inv_valid and cfg.method.repair.enabled:
    #
    # [NEW CODE]:
    # If failed and repair enabled, try one repair
    if not inv_valid and cfg.run.method.repair.enabled:
        repair_prompt = C3OT_REPAIR_PROMPT.format(
            error_message=inv_error,
            question=question,
            commitments=json.dumps(commitments, indent=2),
            previous_answer=final_answer,
        )
        repair_response = llm(repair_prompt)

        results["passes"].append(
            {
                "pass_type": "repair",
                "response": repair_response["response"],
                "tokens": repair_response["tokens"],
                "cost_usd": repair_response["cost_usd"],
            }
        )

        # Extract repaired answer
        repaired_answer = checker.extract_final_answer(repair_response["response"])
        if repaired_answer is not None:
            final_answer = repaired_answer
            results["final_answer"] = final_answer
            results["repaired"] = True

    # Check correctness
    if final_answer is not None:
        results["correct"] = abs(final_answer - ground_truth) < 1e-6
    else:
        results["correct"] = False

    # Calculate total tokens and cost
    results["total_tokens"] = sum(p["tokens"] for p in results["passes"])
    results["total_cost_usd"] = sum(p["cost_usd"] for p in results["passes"])

    return results


def run_baseline_cot_inference(
    example: Dict, llm: LLMInference, cfg, mode: str = "main"
) -> Dict:
    """
    Run baseline CoT inference on one example.

    Args:
        example: Dict with 'question' and 'ground_truth'
        llm: LLM inference client
        cfg: Hydra config
        mode: Execution mode

    Returns:
        Dict with results
    """
    checker = CommitmentChecker()
    question = example["question"]
    ground_truth = example["ground_truth"]

    results = {
        "idx": example["idx"],
        "question": question,
        "ground_truth": ground_truth,
        "method": "baseline_cot",
        "passes": [],
    }

    # Single pass
    prompt = BASELINE_COT_PROMPT.format(question=question)
    response = llm(prompt)

    results["passes"].append(
        {
            "pass_type": "single",
            "response": response["response"],
            "tokens": response["tokens"],
            "cost_usd": response["cost_usd"],
        }
    )

    # Extract final answer
    final_answer = checker.extract_final_answer(response["response"])
    results["final_answer"] = final_answer

    # Check correctness
    if final_answer is not None:
        results["correct"] = abs(final_answer - ground_truth) < 1e-6
    else:
        results["correct"] = False

    # Baseline has no commitments or invariants
    results["commitment_valid"] = None
    results["invariants_satisfied"] = None
    results["commitments"] = None

    results["total_tokens"] = response["tokens"]
    results["total_cost_usd"] = response["cost_usd"]

    return results


def run_inference(
    examples: List[Dict], llm: LLMInference, cfg, mode: str
) -> List[Dict]:
    """
    Run inference on all examples based on method configuration.

    Args:
        examples: List of examples to process
        llm: LLM inference client
        cfg: Hydra config
        mode: Execution mode

    Returns:
        List of results
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Accessing cfg.method when it's actually cfg.run.method
    # [CAUSE]: Hydra loads run configs as nested under cfg.run
    # [FIX]: Access cfg.run.method instead of cfg.method
    #
    # [OLD CODE]:
    # method_name = cfg.method.name
    #
    # [NEW CODE]:
    method_name = cfg.run.method.name

    # In sanity_check mode, limit to first N samples
    if mode == "sanity_check":
        examples = examples[:10]  # Use 10 samples for sanity check
        print(f"Sanity check mode: processing {len(examples)} samples")

    results = []
    for i, example in enumerate(examples):
        print(f"Processing example {i + 1}/{len(examples)}: idx={example['idx']}")

        try:
            if method_name == "c3ot":
                result = run_c3ot_inference(example, llm, cfg, mode)
            elif method_name == "baseline_cot":
                result = run_baseline_cot_inference(example, llm, cfg, mode)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            results.append(result)

        except Exception as e:
            print(f"Error processing example {example['idx']}: {e}")
            # Add error result
            results.append(
                {
                    "idx": example["idx"],
                    "question": example["question"],
                    "ground_truth": example["ground_truth"],
                    "method": method_name,
                    "error": str(e),
                    "correct": False,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                }
            )

    return results
