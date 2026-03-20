"""
Benchmark hot forward paths for the RIEnet PyTorch implementation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from rienet_torch.rnn import KerasGRULayer, KerasLSTMLayer
from rienet_torch.trainable_layers import DeepRecurrentLayer, RIEnetLayer


def benchmark_case(fn: Callable[[], None], *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        samples.append(end - start)

    median_s = statistics.median(samples)
    mean_s = statistics.mean(samples)
    return {
        "median_ms": median_s * 1000.0,
        "mean_ms": mean_s * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
        "iterations": iterations,
        "warmup": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    device = torch.device("cpu")

    rnn_inputs = torch.randn(32, 64, 16, device=device)
    rienet_inputs = torch.randn(16, 32, 64, device=device)

    gru = KerasGRULayer(units=32, name="bench_gru").to(device)
    gru(rnn_inputs)

    lstm = KerasLSTMLayer(units=32, name="bench_lstm").to(device)
    lstm(rnn_inputs)

    deep_rnn = DeepRecurrentLayer(
        recurrent_layer_sizes=[32],
        recurrent_model="GRU",
        direction="bidirectional",
        name="bench_deep_rnn",
    ).to(device)
    deep_rnn(rnn_inputs)

    rienet = RIEnetLayer(output_type="all", name="bench_rienet").to(device)
    rienet(rienet_inputs)

    results = {
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "num_threads": torch.get_num_threads(),
        },
        "cases": {
            "keras_gru_forward_eval": benchmark_case(lambda: gru(rnn_inputs, training=False), warmup=args.warmup, iterations=args.iterations),
            "keras_lstm_forward_eval": benchmark_case(lambda: lstm(rnn_inputs, training=False), warmup=args.warmup, iterations=args.iterations),
            "deep_recurrent_gru_bidirectional_eval": benchmark_case(
                lambda: deep_rnn(rnn_inputs, training=False),
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "rienet_all_forward_eval": benchmark_case(
                lambda: rienet(rienet_inputs, training=False),
                warmup=args.warmup,
                iterations=args.iterations,
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
