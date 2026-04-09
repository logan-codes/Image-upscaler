import gradio as gr
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os

# Load model once at startup (not on every call)
def load_model(scale=4):
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=scale
    )
    model_dir = os.path.join(os.environ["HOME"], "weights")
    os.makedirs(model_dir, exist_ok=True)

    upsampler = RealESRGANer(
        scale=scale,
        model_path=f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x{scale}plus.pth",
        model=model,
        model_dir=model_dir,
        half=False  # set True if GPU available
    )
    return upsampler

upsampler_cache = {}

def upscale(image: Image.Image, scale: int):
    if scale not in upsampler_cache:
        upsampler_cache[scale] = load_model(scale)
    
    upsampler = upsampler_cache[scale]
    img_array = np.array(image)
    output, _ = upsampler.enhance(img_array, outscale=scale)
    return Image.fromarray(output)

# Gradio UI
with gr.Blocks(title="AI Image Upscaler") as demo:
    gr.Markdown("## 🔍 AI Image Upscaler\nPowered by Real-ESRGAN")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            scale_choice = gr.Radio(
                choices=[2, 4], value=4, label="Upscale Factor"
            )
            btn = gr.Button("Upscale", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(type="pil", label="Upscaled Output")
    
    btn.click(fn=upscale, inputs=[input_img, scale_choice], outputs=output_img)

demo.launch(server_name="0.0.0.0", server_port=7860, queue=True)