import os
import sys
import requests
from pathlib import Path

import gradio as gr
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer

# Flush logs immediately (important for HF Spaces)
sys.stdout.reconfigure(line_buffering=True)

MODEL_FILENAME = "RealESRGAN_x4plus.pth"
MODEL_SCALE = 4
SUPPORTED_SCALES = (2, 4)


# -------------------------------
# Model Path + Download Handling
# -------------------------------
def resolve_model_path() -> str:
    project_root = Path(__file__).resolve().parent
    model_dir = project_root / "weights"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / MODEL_FILENAME

    print(f"[INFO] Looking for model at: {model_path}")

    if model_path.exists():
        print("[SUCCESS] Model already exists. Skipping download.")
    else:
        print("[INFO] Model not found. Starting download...")

        url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{MODEL_FILENAME}"
        response = requests.get(url, stream=True)

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"[DOWNLOAD] {percent:.2f}%")

        print("[SUCCESS] Model downloaded successfully!")

    return str(model_path)


# -------------------------------
# Load Model
# -------------------------------
def load_model():
    print("[INFO] Initializing model architecture...")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=MODEL_SCALE,
    )

    print("[INFO] Resolving model path...")
    model_path = resolve_model_path()

    print(f"[INFO] Loading model weights from: {model_path}")

    upsampler = RealESRGANer(
        scale=MODEL_SCALE,
        model_path=model_path,
        model=model,
        half=False,  # set True if GPU available
    )

    print("[SUCCESS] Model loaded successfully!")

    return upsampler


# -------------------------------
# Cache Model
# -------------------------------
upsampler_cache = None


def get_upsampler():
    global upsampler_cache
    if upsampler_cache is None:
        print("[INFO] No cached model found. Loading...")
        upsampler_cache = load_model()
    else:
        print("[INFO] Using cached model.")

    return upsampler_cache


# -------------------------------
# Upscale Function
# -------------------------------
def upscale(image: Image.Image, scale: int):
    print("[INFO] Upscale request received")

    if image is None:
        print("[ERROR] No image provided")
        raise gr.Error("Please upload an image first.")

    if scale not in SUPPORTED_SCALES:
        print(f"[ERROR] Unsupported scale: {scale}")
        raise gr.Error(f"Unsupported upscale factor: {scale}")

    print(f"[INFO] Using scale: {scale}")

    upsampler = get_upsampler()

    print("[INFO] Starting image enhancement...")
    output, _ = upsampler.enhance(np.array(image), outscale=scale)

    print("[SUCCESS] Upscaling completed!")

    return Image.fromarray(output)


# -------------------------------
# Gradio UI
# -------------------------------
def build_demo():
    with gr.Blocks(title="AI Image Upscaler") as app:
        gr.Markdown("## 🔍 AI Image Upscaler\nPowered by Real-ESRGAN")

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Input Image")
                scale_choice = gr.Radio(
                    choices=list(SUPPORTED_SCALES),
                    value=4,
                    label="Upscale Factor",
                )
                btn = gr.Button("Upscale", variant="primary")

            with gr.Column():
                output_img = gr.Image(type="pil", label="Upscaled Output")

        btn.click(fn=upscale, inputs=[input_img, scale_choice], outputs=output_img)

    return app


# -------------------------------
# Run App
# -------------------------------
demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        queue=True  # prevents crashes under load
    )