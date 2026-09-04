#!/usr/bin/env python3
"""Analyze an MP4 segment with a Qwen3.5 VLM served by OVMS.

This mirrors the app's deep analyzer at the MP4/frame-sampling boundary, then
uses OVMS's OpenAI-compatible VLM endpoint by sending sampled frames as ordered
base64 JPEG images.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import av
import requests
from PIL import Image


DEFAULT_PROMPT = (
    "These frames are sampled chronologically from one continuous video segment. "
    "In 2-3 sentences, describe what happens and how it resolves by the end. "
    "Do not refer to individual frames or images. Do not speculate."
)


def _sample_indices(total_frames: int, max_frames: int) -> set[int]:
    sample_count = min(max_frames, total_frames)
    return {min(total_frames - 1, int(index * total_frames / sample_count)) for index in range(sample_count)}


def _frame_to_jpeg_data_url(frame: av.VideoFrame, *, max_edge: int, quality: int) -> str:
    image = frame.to_image().convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def sample_segment_frame_images(
    segment_path: Path,
    *,
    max_frames: int,
    max_edge: int,
    jpeg_quality: int,
) -> list[str]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    container = av.open(str(segment_path))
    try:
        stream = container.streams.video[0]
        total_frames = stream.frames or 0

        if total_frames > 0:
            wanted = _sample_indices(total_frames, max_frames)
            images: list[str] = []
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in wanted:
                    images.append(_frame_to_jpeg_data_url(frame, max_edge=max_edge, quality=jpeg_quality))
            return images

        decoded = [frame for frame in container.decode(stream)]
        if not decoded:
            raise ValueError(f"No frames could be decoded from segment: {segment_path}")
        wanted = _sample_indices(len(decoded), max_frames)
        return [
            _frame_to_jpeg_data_url(frame, max_edge=max_edge, quality=jpeg_quality)
            for frame_index, frame in enumerate(decoded)
            if frame_index in wanted
        ]
    finally:
        container.close()


def build_chat_payload(
    *,
    model: str,
    prompt: str,
    frame_images: list[str],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{prompt}\n\nFrame count: {len(frame_images)}. Treat the images as chronological frames.",
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in frame_images
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def post_chat_completion(endpoint: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    url = endpoint.rstrip("/") + "/chat/completions"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_message_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    return str(content).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OVMS Qwen3.5 deep analysis for one MP4 segment.")
    parser.add_argument("segment", type=Path, help="Path to an MP4 segment to analyze.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v3", help="OVMS v3 endpoint base URL.")
    parser.add_argument("--model", default="qwen3_5_deep_analyzer", help="OVMS model name.")
    parser.add_argument("--frames", type=int, default=8, help="Maximum frames to sample from the segment.")
    parser.add_argument("--max-edge", type=int, default=768, help="Maximum encoded JPEG width or height.")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for sampled frames.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--timeout", type=float, default=300.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent with the sampled frames.")
    parser.add_argument("--output", type=Path, help="Optional path for full JSON result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.segment.is_file():
        raise FileNotFoundError(f"MP4 segment not found: {args.segment}")

    started = time.perf_counter()
    frame_images = sample_segment_frame_images(
        args.segment,
        max_frames=args.frames,
        max_edge=args.max_edge,
        jpeg_quality=args.jpeg_quality,
    )
    payload = build_chat_payload(
        model=args.model,
        prompt=args.prompt,
        frame_images=frame_images,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    inference_started = time.perf_counter()
    response_json = post_chat_completion(args.endpoint, payload, timeout=args.timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    inference_ms = (time.perf_counter() - inference_started) * 1000.0

    result = {
        "segment": str(args.segment),
        "model": args.model,
        "endpoint": args.endpoint,
        "frames_sampled": len(frame_images),
        "total_duration_ms": elapsed_ms,
        "ovms_request_duration_ms": inference_ms,
        "description": extract_message_text(response_json),
        "response": response_json,
    }

    print(json.dumps({key: value for key, value in result.items() if key != "response"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())