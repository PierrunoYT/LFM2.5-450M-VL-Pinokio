import os
from typing import Generator, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

model = None
processor = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _model_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_models() -> None:
    global model, processor
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=_model_dtype(),
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if hasattr(image_processor, "min_image_tokens"):
            image_processor.min_image_tokens = 64
        if hasattr(image_processor, "max_image_tokens"):
            image_processor.max_image_tokens = 256
        if hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = True

    print("Model and processor loaded.")


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------


def _sample_video_frames(video_path: str, max_frames: int) -> List[Image.Image]:
    """Evenly sample up to *max_frames* RGB PIL images from a video file."""
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: List[Image.Image] = []

        if total <= 0:
            # Unknown length — read all frames then subsample.
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if not frames:
                raise ValueError("No frames could be read from the video.")
            if len(frames) > max_frames:
                idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                frames = [frames[int(i)] for i in idxs]
            return frames

        n_take = min(max_frames, total)
        indices = set(np.unique(np.linspace(0, total - 1, n_take, dtype=int)).tolist())
        for pos in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if pos in indices:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(frames) == n_take:
                break

        if not frames:
            raise ValueError("No frames could be read from the video.")
        return frames
    finally:
        cap.release()


def _video_filepath(val: object) -> Optional[str]:
    """Extract a file path string from whatever Gradio passes for a video component."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip():
        return val.strip()
    if isinstance(val, dict):
        for key in ("name", "path", "video"):
            v = val.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def _resolve_image(image_path: Optional[str], image_url: str) -> Image.Image:
    if image_path:
        return load_image(image_path)
    if image_url and image_url.strip():
        return load_image(image_url.strip())
    raise ValueError("Provide an uploaded image or a valid image URL.")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _build_conversation(images: List[Image.Image], user_text: str) -> list:
    content: List[dict] = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": user_text})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful multimodal assistant by Liquid AI.",
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def _generate_reply(
    images: List[Image.Image],
    user_text: str,
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    repetition_penalty: float,
) -> str:
    assert model is not None and processor is not None

    conversation = _build_conversation(images, user_text)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    generate_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": temperature > 0.0,
        "temperature": float(temperature),
        "min_p": float(min_p),
        "repetition_penalty": float(repetition_penalty),
    }

    input_len = inputs["input_ids"].shape[-1]
    try:
        outputs = model.generate(**inputs, **generate_kwargs)
    except TypeError:
        # Older transformers versions may not support min_p.
        generate_kwargs.pop("min_p", None)
        outputs = model.generate(**inputs, **generate_kwargs)

    return processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[
        0
    ].strip()


# ---------------------------------------------------------------------------
# Main chat handler (streaming generator)
# ---------------------------------------------------------------------------


def run_vision_chat(
    video_path: Optional[str],
    max_video_frames: int,
    image_path: Optional[str],
    image_url: str,
    prompt: str,
    stream_each_frame: bool,
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    repetition_penalty: float,
) -> Generator[Tuple[str, str], None, None]:
    if model is None or processor is None:
        yield "", "Model is not loaded yet."
        return
    if not prompt or not prompt.strip():
        yield "", "Please provide a text prompt."
        return

    gen_args = (max_new_tokens, temperature, min_p, repetition_penalty)

    try:
        video_file = _video_filepath(video_path)

        # --- Video input ---
        if video_file:
            images = _sample_video_frames(video_file, int(max_video_frames))
            n = len(images)
            base_prompt = prompt.strip()

            if stream_each_frame:
                # Analyse each frame independently and stream results live.
                accumulated = ""
                for i, frame in enumerate(images):
                    frame_prompt = (
                        f"This is frame {i + 1} of {n} from the same video "
                        f"(chronological order). {base_prompt}"
                    )
                    reply = _generate_reply([frame], frame_prompt, *gen_args)
                    accumulated += f"### Frame {i + 1} / {n}\n{reply}\n\n"
                    yield accumulated, f"Streamed frame {i + 1} / {n}."
                return

            # All frames in a single prompt.
            user_text = base_prompt
            if n > 1:
                user_text = (
                    "These images are frames from one video in chronological order. "
                    + user_text
                )
            reply = _generate_reply(images, user_text, *gen_args)
            yield reply, f"Done. One answer from {n} frame(s) in a single prompt."
            return

        # --- Image input ---
        if image_path or (image_url and image_url.strip()):
            image = _resolve_image(image_path, image_url)
            reply = _generate_reply([image], prompt.strip(), *gen_args)
            yield reply, "Done. Single image."
            return

        yield "", "Provide a video file, an image upload, or an image URL."

    except Exception as exc:
        yield "", f"Error: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

with gr.Blocks(title="LFM2.5-VL-450M") as demo:
    gr.Markdown(
        """
        # LFM2.5-VL-450M
        Compact multimodal model by Liquid AI for image and video understanding.
        Upload a **video** or an **image** (or paste an image URL), type a prompt, and click **Run**.
        For video you can choose a **single combined answer** or **stream one reply per sampled frame**
        so output appears live as each frame is processed.
        """
    )

    with gr.Row():
        # ---- Left column: inputs ----
        with gr.Column():
            video_input = gr.Video(label="Video upload (optional)", sources=["upload"])
            max_video_frames_input = gr.Slider(
                minimum=1,
                maximum=24,
                value=8,
                step=1,
                label="Max frames from video",
            )
            stream_frames_input = gr.Checkbox(
                label="Stream output per frame (one generation per frame; ignored for images)",
                value=False,
            )
            image_input = gr.Image(
                type="filepath",
                label="Image upload (if no video)",
                sources=["upload", "clipboard"],
            )
            image_url_input = gr.Textbox(
                value="",
                label="Image URL (if no video or image file)",
                placeholder="https://example.com/image.jpg",
            )
            prompt_input = gr.Textbox(
                value="What is in this image?",
                label="Prompt",
                lines=3,
            )
            with gr.Accordion("Advanced generation settings", open=False):
                max_new_tokens_input = gr.Slider(
                    minimum=16,
                    maximum=1024,
                    value=256,
                    step=1,
                    label="max_new_tokens",
                )
                temperature_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.1,
                    step=0.05,
                    label="temperature",
                )
                min_p_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.15,
                    step=0.01,
                    label="min_p",
                )
                repetition_penalty_input = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.05,
                    step=0.01,
                    label="repetition_penalty",
                )
            run_button = gr.Button("Run Vision Chat", variant="primary")

        # ---- Right column: outputs ----
        with gr.Column():
            output_text = gr.Textbox(label="Model output", lines=14)
            status_text = gr.Textbox(label="Status", lines=2)

    run_button.click(
        fn=run_vision_chat,
        inputs=[
            video_input,
            max_video_frames_input,
            image_input,
            image_url_input,
            prompt_input,
            stream_frames_input,
            max_new_tokens_input,
            temperature_input,
            min_p_input,
            repetition_penalty_input,
        ],
        outputs=[output_text, status_text],
        show_progress=True,
        api_name="generate",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_models()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    favicon = next(
        (
            p
            for p in (
                os.path.join(here, "icon.png"),
                os.path.join(here, "icon.jpeg"),
                os.path.join(here, "icon.jpg"),
                os.path.join(root, "icon.png"),
                os.path.join(root, "icon.jpeg"),
                os.path.join(root, "icon.jpg"),
            )
            if os.path.isfile(p)
        ),
        None,
    )

    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        favicon_path=favicon,
        theme=_THEME,
    )
