import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

@st.cache_resource
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        revision="fp16" if device == "cuda" else None,
    ).to(device)

    return pipe

st.title("🎨 AI Style Transfer")
st.write("Upload an image and select a style to apply:")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

style_options = [
    "Style",
    "Pixar Art Style",
    "Lego Art Style",
    "Watercolor Art Style",
    "Simpson Art Style",
    "Disney Art Style",
    "Chibi Art Style",
    "Comic Art Style",
    "Anime Art Style",
    "PowerPuff Girls Art Style"
]

selected_style = st.selectbox("Select art style", style_options)

strength = st.slider("Stylization Strength (higher = more styled)", 0.3, 1.0, 0.75, 0.05)

if uploaded_file and selected_style != "Style":
    with st.spinner("Applying style..."):
        input_image = Image.open(uploaded_file).convert("RGB")
        input_image = input_image.resize((512, 512))

        pipe = load_pipeline()

        # Build prompt dynamically
        prompt = f"in the style of {selected_style}"

        result = pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            guidance_scale=7.5,
        ).images[0]

        st.image(result, caption=f"Image in '{selected_style}'", use_column_width=True)
elif uploaded_file:
    st.warning("Please select an art style from the dropdown.")
else:
    st.info("Please upload an image and select a style to begin.")
