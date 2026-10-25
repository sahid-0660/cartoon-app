import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

st.set_page_config(page_title="Cartoon Photo Maker HD", page_icon="🎨")
st.title("🎨 Cartoon Photo Maker - HD")
st.write("Photo de, style beche ne!")

style = st.selectbox("Style beche ne:", [
    "Ghibli",
    "Normal Cartoon",
    "Pixar / 3D",
    "Anime",
    "Pencil Sketch",
    "Pop Art"
])

smooth = st.slider("Smooth / Soft kotota hobe:", 1, 10, 6)
color_strong = st.slider("Color kotota deep hobe:", 1, 20, 13)

uploaded = st.file_uploader("Tor photo de", type=["jpg","jpeg","png"])

def to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def make_hd(pil_img):
    # 1.5x boro kore quality barano
    w, h = pil_img.size
    return pil_img.resize((int(w*1.5), int(h*1.5)), Image.LANCZOS)

def apply_color(img, s):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
    hsv[:,:,1] = hsv[:,:,1] * (s/10)
    hsv[:,:,2] = hsv[:,:,2] * 1.08
    hsv = np.clip(hsv,0,255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def cartoon_ghibli(img, smooth, s):
    # face bachanor jonno halka smooth
    out = cv2.edgePreservingFilter(img, flags=1, sigma_s=40+smooth*5, sigma_r=0.35)
    out = cv2.detailEnhance(out, sigma_s=8, sigma_r=0.12)
    out = apply_color(out, s)
    # halka bright
    out = cv2.convertScaleAbs(out, alpha=1.05, beta=8)
    return out

def cartoon_normal(img, smooth, s):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 200, 200)
    color = apply_color(color, s)
    return cv2.bitwise_and(color, color, mask=edges)

def cartoon_pixar(img, smooth, s):
    out = cv2.bilateralFilter(img, 15, 80, 80)
    out = cv2.bilateralFilter(out, 15, 80, 80)
    out = apply_color(out, s+3)
    out = cv2.convertScaleAbs(out, alpha=1.08, beta=10)
    return out

def cartoon_anime(img, smooth, s):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(img, 9, 200, 200)
    color = (color//28)*28
    color = apply_color(color, s)
    return cv2.bitwise_and(color, color, mask=edges)

def cartoon_sketch(img, smooth, s):
    gray, _ = cv2.pencilSketch(img, sigma_s=50, sigma_r=0.07, shade_factor=0.06)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def cartoon_pop(img, smooth, s):
    color = cv2.bilateralFilter(img, 9, 200, 200)
    color = (color//35)*35
    return apply_color(color, s+5)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Asol Photo", use_column_width=True)
    cv_img = to_cv(image)

    if style == "Ghibli":
        out = cartoon_ghibli(cv_img, smooth, color_strong)
    elif style == "Normal Cartoon":
        out = cartoon_normal(cv_img, smooth, color_strong)
    elif style == "Pixar / 3D":
        out = cartoon_pixar(cv_img, smooth, color_strong)
    elif style == "Anime":
        out = cartoon_anime(cv_img, smooth, color_strong)
    elif style == "Pencil Sketch":
        out = cartoon_sketch(cv_img, smooth, color_strong)
    else:
        out = cartoon_pop(cv_img, smooth, color_strong)

    result = to_pil(out)
    result = make_hd(result)
    # last e sharp
    result = ImageEnhance.Sharpness(result).enhance(1.4)
    result = ImageEnhance.Color(result).enhance(1.15)

    st.image(result, caption=f"{style} - HD", use_column_width=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button("⬇️ HD Download", buf.getvalue(), file_name=f"{style}_HD.png", mime="image/png")
