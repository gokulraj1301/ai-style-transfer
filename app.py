import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import os

# Load model
@st.cache_resource
def load_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=False
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

pipe = load_pipeline()

# Prompts
style_prompts = {
    "Ghibli": "A portrait in Studio Ghibli anime style",
    "Pixar": "Pixar-style 3D render of a person",
    "Lego": "Lego figure style image of a person",
    "Watercolor": "Watercolor painting of a person",
    "Simpson": "A character in the Simpsons cartoon style",
    "Disney": "Disney animated character portrait",
    "Chibi": "Cute Chibi anime character portrait",
    "Comic": "Comic book style portrait illustration",
    "Anime": "Japanese anime style character",
    "Powerpuff": "Powerpuff Girls character portrait"
}

st.title("🎨 AI Art Style Transfer")
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    input_image = input_image.resize((512, 512))
    st.image(input_image, caption="Original Image")

    style = st.selectbox("Choose Art Style", list(style_prompts.keys()))
    if st.button("Generate Art Style"):
        with st.spinner("Generating..."):
            prompt = style_prompts[style]
            result = pipe(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images[0]
            result.save("stylized.png")
            st.image(result, caption=f"{style} Style")
            with open("stylized.png", "rb") as f:
                st.download_button(label="Download PNG", data=f, file_name=f"{style.lower()}_styled.png", mime="image/png")
