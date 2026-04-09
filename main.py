import os
from pathlib import Path

import gradio as gr
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer

MODEL_FILENAME = "RealESRGAN_x4plus.pth"
MODEL_SCALE = 4
SUPPORTED_SCALES = (2, 4)


def resolve_model_path() -> str:
    project_root = Path(__file__).resolve().parent
    candidate_paths = [project_root / "weights" / MODEL_FILENAME]

    home_dir = os.environ.get("HOME")
    if home_dir:
        candidate_paths.append(Path(home_dir) / "weights" / MODEL_FILENAME)

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    checked_locations = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not find {MODEL_FILENAME}. Checked: {checked_locations}"
    )


def load_model():
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=MODEL_SCALE,
    )
    return RealESRGANer(
        scale=MODEL_SCALE,
        model_path=resolve_model_path(),
        model=model,
        half=False,
    )


upsampler_cache = None


def get_upsampler():
    global upsampler_cache
    if upsampler_cache is None:
        upsampler_cache = load_model()
    return upsampler_cache


def upscale(image: Image.Image, scale: int):
    if image is None:
        raise gr.Error("Please upload an image first.")

    if scale not in SUPPORTED_SCALES:
        raise gr.Error(f"Unsupported upscale factor: {scale}")

    upsampler = get_upsampler()
    output, _ = upsampler.enhance(np.array(image), outscale=scale)
    return Image.fromarray(output)


def build_demo():
    with gr.Blocks(title="AI Image Upscaler") as app:
        gr.Markdown("## 🔍 AI Image Upscaler\nPowered by Real-ESRGAN")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Input Image")
                scale_choice = gr.Radio(
                    choices=list(SUPPORTED_SCALES), value=4, label="Upscale Factor"
                )
                btn = gr.Button("Upscale", variant="primary")

            with gr.Column():
                output_img = gr.Image(type="pil", label="Upscaled Output")

        btn.click(fn=upscale, inputs=[input_img, scale_choice], outputs=output_img)
    return app


demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
