# OVMS Deep Analyzer PoC

Standalone proof of concept for analyzing one MP4 segment with the Qwen3.5 VLM served by OpenVINO Model Server.

The production deep analyzer passes sampled segment frames to `openvino_genai.VLMPipeline.generate(..., videos=[tensor])`. Stock OVMS exposes VLMs through OpenAI-compatible REST APIs, so this PoC samples the MP4 in the same spirit, encodes those frames as chronological JPEG image inputs, and sends them to `/v3/chat/completions`.

## Start OVMS

From this directory:

```bash
docker compose up -d
```

Defaults:

- image: `openvino/model_server:latest`
- model path: `../../ov_models/cpu/Qwen3.5-2B-int4-ov`
- target device: `CPU`
- REST endpoint: `http://127.0.0.1:8000/v3`
- served model name: `qwen3_5_deep_analyzer`

Qwen3.5 model cards require OpenVINO 2026.2 or newer. If your local `latest` tag is older, set an explicit OVMS image tag that contains OpenVINO 2026.2+.

GPU run, if OpenVINO sees a supported GPU inside the OVMS container:

```bash
POC_MODEL_DIR=../../ov_models/gpu/Qwen3.5-2B-int4-ov \
OVMS_TARGET_DEVICE=GPU \
docker compose up -d
```

Check readiness:

```bash
curl http://localhost:8000/v3/models
```

## Run MP4 Analysis

Create a local environment and install only the OVMS client dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Analyze a segment:

```bash
python analyze_mp4.py ../../segments/<stream-id>/<segment>.mp4 \
  --frames 8 \
  --max-tokens 128 \
  --output result.json
```

The script prints a compact result and optionally writes the full OVMS response.

## Run Native GenAI Video Tensor Analysis

Use this companion path to compare OVMS ordered-image input against the current deep analyzer style:

```bash
pip install -r requirements-genai.txt
python analyze_mp4_genai.py ../../segments/<stream-id>/<segment>.mp4 \
  --frames 8 \
  --max-tokens 128 \
  --output genai-result.json
```

This script keeps the production-style boundary:

```text
MP4 -> sampled frames -> ov.Tensor(frames) -> VLMPipeline.generate(..., videos=[tensor])
```

The OVMS script keeps the serving-style boundary:

```text
MP4 -> sampled frames -> JPEG/base64 images -> OVMS /v3/chat/completions
```

By default, `analyze_mp4_genai.py` uses `bgr24` frames to mirror `app/backend/services/deep_analyzer.py`. You can test RGB input with `--frame-format rgb24` if you want to compare model sensitivity to channel order.

## Decision Criteria

Use this PoC to compare against the in-process GenAI deep analyzer:

- Does OVMS load `Qwen3.5-2B-int4-ov` reliably on the target device?
- Does ordered multi-image input preserve enough temporal context for segment summaries?
- Are latency and memory acceptable compared with the existing `videos=[tensor]` path?
- If OVMS rejects multiple images, the next option is a custom OVMS graph or a sidecar GenAI service for exact video tensor semantics.