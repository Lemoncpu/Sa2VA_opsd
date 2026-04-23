import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ("loss_opsd_total", "loss_opsd_jsd", "verifier_iou")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training loss and IoU curves from MMEngine logs."
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to a training .log file, JSONL log file, or work_dir containing logs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <log_dir>/training_metrics.png",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Optional moving average window. Use 1 to disable smoothing.",
    )
    return parser.parse_args()


def find_candidate_files(log_path: Path):
    if log_path.is_file():
        return [log_path]

    if not log_path.exists():
        raise FileNotFoundError(f"Path does not exist: {log_path}")

    candidates = []
    candidates.extend(sorted(log_path.glob("*.json")))
    candidates.extend(sorted(log_path.glob("*.jsonl")))
    candidates.extend(sorted(log_path.glob("*.log")))

    vis_data = log_path / "vis_data"
    if vis_data.exists():
        candidates.extend(sorted(vis_data.glob("*.json")))
        candidates.extend(sorted(vis_data.glob("*.jsonl")))

    if not candidates:
        raise FileNotFoundError(f"No log files found under: {log_path}")
    return candidates


def parse_json_lines(path: Path):
    metric_data = {metric: [] for metric in METRICS}

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            step = record.get("iter", record.get("step", record.get("iteration")))
            if step is None:
                continue

            for metric in METRICS:
                value = record.get(metric)
                if isinstance(value, (int, float)):
                    metric_data[metric].append((int(step), float(value)))

    return metric_data


def parse_plain_json(path: Path):
    metric_data = {metric: [] for metric in METRICS}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return metric_data

    if isinstance(payload, list):
        for record in payload:
            if not isinstance(record, dict):
                continue
            step = record.get("iter", record.get("step", record.get("iteration")))
            if step is None:
                continue
            for metric in METRICS:
                value = record.get(metric)
                if isinstance(value, (int, float)):
                    metric_data[metric].append((int(step), float(value)))

    return metric_data


def parse_log_text(path: Path):
    metric_data = {metric: [] for metric in METRICS}
    iter_regex = re.compile(r"\b(?:Iter|iter)[^\d]*(\d+)\b")
    metric_regexes = {
        metric: re.compile(rf"\b{re.escape(metric)}:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
        for metric in METRICS
    }

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            iter_match = iter_regex.search(line)
            if iter_match is None:
                continue
            step = int(iter_match.group(1))
            for metric, regex in metric_regexes.items():
                metric_match = regex.search(line)
                if metric_match is not None:
                    metric_data[metric].append((step, float(metric_match.group(1))))

    return metric_data


def merge_metric_data(metric_groups):
    merged = {metric: [] for metric in METRICS}
    for metric_data in metric_groups:
        for metric in METRICS:
            merged[metric].extend(metric_data.get(metric, []))

    for metric in METRICS:
        merged[metric] = sorted(set(merged[metric]), key=lambda item: item[0])
    return merged


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    smoothed = []
    running = 0.0
    queue = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        smoothed.append(running / len(queue))
    return smoothed


def choose_parser(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".jsonl"}:
        return parse_json_lines(path)
    if suffix in {".json"}:
        data = parse_json_lines(path)
        if any(data[metric] for metric in METRICS):
            return data
        return parse_plain_json(path)
    return parse_log_text(path)


def plot_metrics(metric_data, output_path: Path, smooth: int):
    available_metrics = [metric for metric in METRICS if metric_data[metric]]
    if not available_metrics:
        raise ValueError("No metric data found for loss_opsd_total/loss_opsd_jsd or verifier_iou.")

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 4 * len(available_metrics)), sharex=True)
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        steps = [item[0] for item in metric_data[metric]]
        values = [item[1] for item in metric_data[metric]]
        smooth_values = moving_average(values, smooth)

        ax.plot(steps, values, alpha=0.35, linewidth=1.0, label=f"{metric} (raw)")
        ax.plot(steps, smooth_values, linewidth=2.0, label=f"{metric} (smooth={smooth})")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Training Metrics")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    log_path = Path(args.log_path)
    candidate_files = find_candidate_files(log_path)
    metric_groups = [choose_parser(path) for path in candidate_files]
    metric_data = merge_metric_data(metric_groups)

    if args.output is not None:
        output_path = Path(args.output)
    else:
        base_dir = log_path if log_path.is_dir() else log_path.parent
        output_path = base_dir / "training_metrics.png"

    plot_metrics(metric_data, output_path, smooth=max(1, args.smooth))
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
