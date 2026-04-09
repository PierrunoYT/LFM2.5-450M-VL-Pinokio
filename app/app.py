import os
from typing import List, Optional, Tuple

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


def _model_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


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


def _sample_video_frames(video_path: str, max_frames: int) -> List[Image.Image]:
    """Evenly sample up to max_frames from a video (RGB PIL images)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_out: List[Image.Image] = []

    try:
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1.")

        if total <= 0:
            while len(frames_out) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_out.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if not frames_out:
                raise ValueError("No frames could be read from the video.")
            if len(frames_out) > max_frames:
                idxs = np.linspace(0, len(frames_out) - 1, max_frames, dtype=int)
                frames_out = [frames_out[int(i)] for i in idxs]
            return frames_out

        n_take = min(max_frames, total)
        indices = np.unique(np.linspace(0, total - 1, n_take, dtype=int))
        needed = set(indices.tolist())
        seen = 0
        while seen < total and len(frames_out) < len(indices):
            ok, frame = cap.read()
            if not ok:
                break
            if seen in needed:
                frames_out.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            seen += 1

        if not frames_out:
            raise ValueError("No frames could be read from the video.")
        return frames_out
    finally:
        cap.release()


def _video_filepath(val: object) -> Optional[str]:
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


def _resolve_image(image_path: Optional[str], image_url: str) -> Image.Image:
    if image_path:
        return load_image(image_path)
    if image_url and image_url.strip():
        return load_image(image_url.strip())
    raise ValueError("Provide an uploaded image or a valid image URL.")


def run_vision_chat(
    video_path: Optional[str],
    max_video_frames: int,
    image_path: Optional[str],
    image_url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    repetition_penalty: float,
) -> Tuple[str, str]:
    if model is None or processor is None:
        return "", "Model is not loaded yet."
    if not prompt or not prompt.strip():
        return "", "Please provide a text prompt."

    try:
        images: List[Image.Image] = []
        mode_note = ""

        video_file = _video_filepath(video_path)
        if video_file:
            images = _sample_video_frames(video_file, int(max_video_frames))
            mode_note = f"Using {len(images)} sampled video frame(s)."
        elif image_path or (image_url and image_url.strip()):
            images = [_resolve_image(image_path, image_url)]
            mode_note = "Using a single image."
        else:
            return "", "Provide a video file, an image upload, or an image URL."

        user_text = prompt.strip()
        if video_file and len(images) > 1:
            user_text = (
                "These images are frames from one video in chronological order. "
                + user_text
            )

        content: List[dict] = []
        for im in images:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": user_text})

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful multimodal assistant by Liquid AI.",
                    }
                ],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

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

        try:
            outputs = model.generate(**inputs, **generate_kwargs)
        except TypeError:
            generate_kwargs.pop("min_p", None)
            outputs = model.generate(**inputs, **generate_kwargs)

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return decoded, f"Done. {mode_note}"
    except Exception as exc:
        return "", f"Error: {exc}"


_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


with gr.Blocks(
    title="LFM2.5-VL-450M",
    theme=_APP_THEME,
) as demo:
    gr.Markdown(
        """
        # LFM2.5-VL-450M
        Compact multimodal model by Liquid AI for image and video understanding.
        Upload **video** or an **image** (or image URL), ask a question, and generate a response.
        Video uses evenly spaced frames as multiple images (same pathway as Liquid’s streaming demos).
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="Video upload (optional)",
                sources=["upload"],
            )
            max_video_frames_input = gr.Slider(
                minimum=2,
                maximum=24,
                value=8,
                step=1,
                label="Max frames from video",
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
            with gr.Accordion("Advanced Generation Settings", open=False):
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

        with gr.Column():
            output_text = gr.Textbox(label="Model Output", lines=14)
            status_text = gr.Textbox(label="Status", lines=2)

    run_button.click(
        fn=run_vision_chat,
        inputs=[
            video_input,
            max_video_frames_input,
            image_input,
            image_url_input,
            prompt_input,
            max_new_tokens_input,
            temperature_input,
            min_p_input,
            repetition_penalty_input,
        ],
        outputs=[output_text, status_text],
        show_progress=True,
        api_name="generate",
    )


if __name__ == "__main__":
    load_models()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    favicon = None
    for candidate in (
        os.path.join(here, "icon.png"),
        os.path.join(here, "icon.jpeg"),
        os.path.join(here, "icon.jpg"),
        os.path.join(root, "icon.jpeg"),
        os.path.join(root, "icon.jpg"),
        os.path.join(root, "icon.png"),
    ):
        if os.path.isfile(candidate):
            favicon = candidate
            break

    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        favicon_path=favicon,
    )
