import os
import base64
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load API key from .env
load_dotenv()
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
VISION_MODEL = "accounts/fireworks/models/qwen3p7-plus"  # vision-capable LLM


def describe_image(image_bytes: bytes, prompt: str, api_key: str) -> str:
    """
    Send image + prompt to Fireworks vision model.
    Returns the LLM's text description.
    """
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        "max_tokens": 2048,
        "temperature": 0.4,
    }
    r = requests.post(
        FIREWORKS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )

    # Handle common errors gracefully (per recruiter spec)
    if r.status_code == 401:
        raise RuntimeError("Invalid or missing Fireworks API key (HTTP 401).")
    if r.status_code == 429:
        raise RuntimeError("Fireworks rate limit or insufficient credit (HTTP 429).")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ───────── Streamlit UI ─────────
st.set_page_config(page_title="AI Image Describer", page_icon="📷")
st.title("📷 AI Image Describer")
st.caption("Upload an image and let the vision LLM describe it.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
prompt = st.text_input(
    "Custom prompt (optional)",
    value="Describe what is in this image in detail.",
)

if st.button("Describe") and uploaded is not None:
    if not FIREWORKS_API_KEY:
        st.error("FIREWORKS_API_KEY is missing. Add it to your .env file.")
    else:
        # Show the uploaded image
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            try:
                description = describe_image(
                    uploaded.getvalue(), prompt, FIREWORKS_API_KEY
                )
                st.subheader("Description")
                st.write(description)
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")