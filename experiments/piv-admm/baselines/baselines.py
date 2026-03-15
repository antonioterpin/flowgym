"""Script to replicate the baselines in https://arxiv.org/abs/2512.11695.

To run it:
    uv run python experiments/piv-admm/baselines/baselines.py

Results are stored in results/piv_admm_baselines.csv

You can select which GPU to use by setting the CUDA_VISIBLE_DEVICES environment variable.
To run this experiment, we suggest to first run 'uv sync --all-groups' to install all dependencies.
"""

import csv
import os
import re
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

# === SETTINGS ===
ALGORITHMS_PATH = Path("experiments/piv-admm/baselines/algorithms/")
CSV_PATH = Path("results/piv_admm_baselines.csv")

RUN_CMD = [
    "python",
    "-m",
    "src.main",
    "--model",
    "PLACEHOLDER",
    "--dataset",
    "experiments/piv-admm/baselines/val_dataset.yaml",
    "--mode",
    "eval",
]

CUDA_VISIBLE_DEVICES = "0"
TAIL_LINES = 50  # how many final lines to parse for metrics
# =================

# Regex patterns for metrics
PATTERNS = {
    "mean_epe": re.compile(r"Mean EPE.*: ([\d.]+)"),
    "max_epe": re.compile(r"Max EPE.*: ([\d.]+)"),
    "min_epe": re.compile(r"Min EPE.*: ([\d.]+)"),
    "mean_rel_epe": re.compile(r"Mean Relative EPE.*: ([\d.]+)"),
    "max_rel_epe": re.compile(r"Max Relative EPE.*: ([\d.]+)"),
    "min_rel_epe": re.compile(r"Min Relative EPE.*: ([\d.]+)"),
    "time": re.compile(r"Total evaluation time: ([\d.]+(?:\.\d+)?)"),
}


def extract_metrics(lines: list[str]) -> dict[str, float]:
    """Extract the required metrics from a list of lines.

    Args:
        lines: List of output lines to parse for metrics.

    Returns:
        Dictionary mapping metric names to their float values.
    """
    metrics = {}
    for line in lines:
        for key, pattern in PATTERNS.items():
            m = pattern.search(line)
            if m:
                metrics[key] = float(m.group(1))
    return metrics


def run_baselines():
    """Run the parameter sweep by modifying the config and launching experiments."""
    # Prepare environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    # Prepare CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(
                [
                    "algorithm",
                    "mean_epe",
                    "max_epe",
                    "min_epe",
                    "mean_rel_epe",
                    "max_rel_epe",
                    "min_rel_epe",
                    "time",
                ]
            )

        # Iterate over yaml configs
        for algo_file in sorted(ALGORITHMS_PATH.glob("*.yaml")):
            name = algo_file.stem
            print(f"\n=== Running {name} ===")

            # Build command
            cmd = RUN_CMD.copy()
            cmd[cmd.index("PLACEHOLDER")] = str(algo_file)

            print(" ".join(cmd))
            print("----------------------------------------")

            # Launch process (live streaming)
            process = Popen(
                cmd, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, env=env
            )

            output_lines = []

            # Stream logs in real time
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")  # Live print
                    output_lines.append(line)

            process.wait()

            print("----------------------------------------")

            # Tail extraction
            tail = output_lines[-TAIL_LINES:]
            metrics = extract_metrics(tail)

            print(f"Extracted metrics for {name}: {metrics}")

            # Write row to CSV
            writer.writerow(
                [
                    name,
                    metrics.get("mean_epe"),
                    metrics.get("max_epe"),
                    metrics.get("min_epe"),
                    metrics.get("mean_rel_epe"),
                    metrics.get("max_rel_epe"),
                    metrics.get("min_rel_epe"),
                    metrics.get("time"),
                ]
            )

            print(f"Saved metrics for {name}.\n")


if __name__ == "__main__":
    run_baselines()
