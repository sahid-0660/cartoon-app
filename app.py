import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Cartoon Photo Maker", page_icon="🎨")
st.title("🎨 Cartoon Photo Maker")
st.write("Tor photo de, style beche ne, cartoon hoye jabe!")

style = st.selectbox("Style beche ne:", [
    "Normal Cartoon",
    "Ghibli",
    "Pixar / 3D",
    "Anime",
    "Pencil Sketch",
    "Pop Art"
])

uploaded = st.file_uploader("Tor photo de", type=["jpg","jpeg","png"])

def to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def cartoon_normal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def cartoon_ghibli(img):
    smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)
    smooth = cv2.detailEnhance(smooth, sigma_s=10, sigma_r=0.15)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(float)
    hsv[:,:,1] = hsv[:,:,1]*1.3
    hsv[:,:,2] = hsv[:,:,2]*1.05
    hsv = np.clip(hsv,0,255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def cartoon_pixar(img):
    smooth = cv2.bilateralFilter(img, 15, 80, 80)
    smooth = cv2.bilateralFilter(smooth, 15, 80, 80)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(float)
    hsv[:,:,1] = hsv[:,:,1]*1.5
    hsv[:,:,2] = hsv[:,:,2]*1.1
    hsv = np.clip(hsv,0,255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def cartoon_anime(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(img, 9, 200, 200)
    color = (color//32)*32
    return cv2.bitwise_and(color, color, mask=edges)

def cartoon_sketch(img):
    gray, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def cartoon_pop(img):
    color = cv2.bilateralFilter(img, 9, 200, 200)
    color = (color//40)*40
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).astype(float)
    hsv[:,:,1] = hsv[:,:,1]*1.8
    hsv = np.clip(hsv,0,255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Asol Photo", use_column_width=True)
    cv_img = to_cv(image)

    if style == "Normal Cartoon":
        out = cartoon_normal(cv_img)
    elif style == "Ghibli":
        out = cartoon_ghibli(cv_img)
    elif style == "Pixar / 3D":
        out = cartoon_pixar(cv_img)
    elif style == "Anime":
        out = cartoon_anime(cv_img)
    elif style == "Pencil Sketch":
        out = cartoon_sketch(cv_img)
    else:
        out = cartoon_pop(cv_img)

    result = to_pil(out)
    st.image(result, caption=f"{style} Style", use_column_width=True)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button("Download Cartoon", buf.getvalue(), file_name=f"{style}.png", mime="image/png")
