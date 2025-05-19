import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

@st.cache_resource
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        revision="fp16" if device == "cuda" else None,
    )
    return pipe.to(device)

st.title("🎨 AI Style Transfer")
st.write("Upload an image and describe the style you'd like to apply!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
style_prompt = st.text_input("Enter style prompt (e.g., 'in the style of Van Gogh')")

if uploaded_file and style_prompt:
    with st.spinner("Applying style..."):
        input_image = Image.open(uploaded_file).convert("RGB")
        input_image = input_image.resize((512, 512))

        # Convert image to latent space if needed (depends on pipeline)
        # Here we simulate style transfer using the prompt as conditioning
        pipe = load_pipeline()
        result = pipe(prompt=style_prompt, image=input_image).images[0]

        st.image(result, caption="Styled Image", use_column_width=True)
else:
    st.info("Please upload an image and enter a style prompt to begin.")
