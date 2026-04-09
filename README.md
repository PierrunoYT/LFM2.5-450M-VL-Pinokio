# LFM2.5-VL-450M (Pinokio)

[Pinokio](https://pinokio.computer/) launcher and **Gradio** front-end for **[LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)** — a compact **vision–language** model for image understanding. The app loads the model with **Hugging Face Transformers** (`AutoModelForImageTextToText` + `AutoProcessor`), uses **Accelerate** `device_map="auto"`, and prefers **bfloat16** on CUDA.

## Features

- **Vision chat**: Upload an image (or paste an **image URL**), enter a text prompt, and generate a response.
- **Advanced generation**: Optional sliders for `max_new_tokens`, `temperature`, `min_p`, and `repetition_penalty` (see **Advanced Generation Settings** in the UI).
- **Local Gradio UI**: Runs on `127.0.0.1` by default; exposes the usual Gradio **REST API** and `/docs` when the server is up.

## Requirements

- **Python**: 3.10+ recommended (Pinokio `torch.js` wheels target **CPython 3.10** on some platforms).
- **GPU**: Strongly recommended (NVIDIA with CUDA). CPU fallback uses `float32` and is much slower.
- **RAM / VRAM**: Depends on Transformers + model weights; allow several GB free.
- **Disk**: Model weights download from Hugging Face on first load unless already cached.

## Repository layout

- `app/` — `app.py` (Gradio UI + inference), `requirements.txt`.
- Root — Pinokio scripts: `pinokio.js`, `install.js`, `start.js`, `update.js`, `reset.js`, `link.js`, `torch.js`.

## Install (manual)

From the repository root:

```bash
pip install -r app/requirements.txt
```

Install a **PyTorch** build that matches your OS and GPU (see [pytorch.org](https://pytorch.org/get-started/locally/)). The Pinokio installer uses `torch.js` to install CUDA builds on NVIDIA Linux/Windows when you install through Pinokio.

Optional: set `HF_TOKEN` if you need authenticated Hub access (model card may list license or access rules).

## Run (manual)

From the repository root:

```bash
python app/app.py
```

Open the URL printed in the terminal (default Gradio port is **7860** unless another process is using it).

## Pinokio

1. Open this repo in Pinokio.
2. **Install** — installs `app/requirements.txt` into the `env` venv and runs `torch.js` for PyTorch.
3. **Start** — runs `python app/app.py` and watches the log for a local `http://...` URL; **Open Web UI** appears when Pinokio captures it.

**Sanity-check launcher scripts** (from repo root):

```bash
node --check install.js start.js pinokio.js update.js reset.js link.js torch.js
```

If **Open Web UI** does not show, check Pinokio / API logs and the terminal output for the Gradio URL.

## API and automation

While the app is running:

- Interactive API: `http://127.0.0.1:7860/docs` (adjust host/port if yours differs).
- OpenAPI JSON: `http://127.0.0.1:7860/openapi.json`.
- The primary Gradio call is **`/generate`** (`api_name="generate"` in `app/app.py`).

## Troubleshooting

- **Out of memory**: Close other GPU apps; try a smaller `max_new_tokens`; ensure no duplicate large models are loaded.
- **Slow first run**: The Hub download and kernel compile/warmup can take time.
- **Windows**: App binds to `127.0.0.1` (see `demo.launch` in `app/app.py`).

## License and credits

- **App code**: There is no `LICENSE` file in the repo root yet; confirm terms before you redistribute or reuse the launcher/app code.
- **Model**: Terms are defined on the [model card](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) (Liquid AI / Hugging Face).
- **Stack**: [Gradio](https://gradio.app/), [Transformers](https://github.com/huggingface/transformers), [Pinokio](https://pinokio.computer/).
