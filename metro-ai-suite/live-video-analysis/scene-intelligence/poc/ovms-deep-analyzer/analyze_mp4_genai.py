#!/usr/bin/env python3
"""Analyze an MP4 segment with the native OpenVINO GenAI video tensor path."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import openvino as ov
import openvino_genai as ov_genai


DEFAULT_PROMPT = (
    "These frames are sampled chronologically from one continuous video segment. "
    "In 2-3 sentences, describe what happens and how it resolves by the end. "
    "Do not refer to individual frames or images. Do not speculate."
)


def _sample_indices(total_frames: int, max_frames: int) -> set[int]:
    sample_count = min(max_frames, total_frames)
    return {min(total_frames - 1, int(index * total_frames / sample_count)) for index in range(sample_count)}


def sample_segment_frames(segment_path: Path, *, max_frames: int, frame_format: str) -> np.ndarray:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    container = av.open(str(segment_path))
    try:
        stream = container.streams.video[0]
        total_frames = stream.frames or 0
        frames: list[np.ndarray] = []

        if total_frames > 0:
            wanted = _sample_indices(total_frames, max_frames)
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in wanted:
                    frames.append(frame.to_ndarray(format=frame_format))
        else:
            decoded = [frame.to_ndarray(format=frame_format) for frame in container.decode(stream)]
            if decoded:
                wanted = _sample_indices(len(decoded), max_frames)
                frames = [frame for frame_index, frame in enumerate(decoded) if frame_index in wanted]
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No frames could be decoded from segment: {segment_path}")
    return np.stack(frames)


def extract_text(result: Any) -> str:
    texts = getattr(result, "texts", None)
    if isinstance(texts, (list, tuple)) and texts:
        return str(texts[0]).strip()
    return str(result).strip()


def extract_perf_metrics(result: Any) -> dict[str, Any]:
    metrics = getattr(result, "perf_metrics", None)
    if metrics is None:
        return {}

    extracted: dict[str, Any] = {}
    for name in dir(metrics):
        if name.startswith("_"):
            continue
        value = getattr(metrics, name, None)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            extracted[name] = value
    return extracted


def analyze_segment(
    *,
    segment_path: Path,
    model_path: Path,
    device: str,
    prompt: str,
    max_frames: int,
    max_tokens: int,
    frame_format: str,
    cache_dir: Path | None,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    frames = sample_segment_frames(segment_path, max_frames=max_frames, frame_format=frame_format)
    tensor = ov.Tensor(frames)

    pipeline_kwargs: dict[str, str] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pipeline_kwargs["CACHE_DIR"] = str(cache_dir)

    load_started = time.perf_counter()
    pipe = ov_genai.VLMPipeline(str(model_path), device, **pipeline_kwargs)
    load_duration_ms = (time.perf_counter() - load_started) * 1000.0

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = max_tokens
    # Greedy decoding loops on repeated phrases without these; see PoC comparison notes.
    generation_config.repetition_penalty = repetition_penalty
    if no_repeat_ngram_size > 0:
        generation_config.no_repeat_ngram_size = no_repeat_ngram_size

    inference_started = time.perf_counter()
    result = pipe.generate(prompt, videos=[tensor], generation_config=generation_config)
    inference_duration_ms = (time.perf_counter() - inference_started) * 1000.0

    return {
        "segment": str(segment_path),
        "model_path": str(model_path),
        "device": device,
        "frames_sampled": int(frames.shape[0]),
        "tensor_shape": str(frames.shape),
        "tensor_dtype": str(frames.dtype),
        "frame_format": frame_format,
        "load_duration_ms": load_duration_ms,
        "inference_duration_ms": inference_duration_ms,
        "description": extract_text(result),
        "perf_metrics": extract_perf_metrics(result),
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native GenAI video-tensor analysis for one MP4 segment.")
    parser.add_argument("segment", type=Path, help="Path to an MP4 segment to analyze.")
    parser.add_argument("--model-path", type=Path, default=Path("../../ov_models/cpu/Qwen3.5-2B-int4-ov"))
    parser.add_argument("--device", default="CPU", help="OpenVINO device for GenAI inference.")
    parser.add_argument("--frames", type=int, default=8, help="Maximum frames to sample from the segment.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--frame-format", default="bgr24", help="PyAV frame format; bgr24 mirrors app deep_analyzer.py.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent with the video tensor.")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "genai_deep_analyzer_poc_cache")
    parser.add_argument("--repetition-penalty", type=float, default=1.3, help="Penalty (>1.0) to discourage repeated phrases.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="Block repeats of this n-gram size; 0 disables.")
    parser.add_argument("--output", type=Path, help="Optional path for full JSON result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.segment.is_file():
        raise FileNotFoundError(f"MP4 segment not found: {args.segment}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")

    result = analyze_segment(
        segment_path=args.segment,
        model_path=args.model_path,
        device=args.device,
        prompt=args.prompt,
        max_frames=args.frames,
        max_tokens=args.max_tokens,
        frame_format=args.frame_format,
        cache_dir=args.cache_dir,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())