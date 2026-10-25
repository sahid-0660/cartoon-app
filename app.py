import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cartoon App")
st.title("Cartoon Photo Maker")

photo = st.file_uploader("Tor photo de", type=["jpg","png","jpeg"])

if photo:
    img = Image.open(photo).convert("RGB")
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(img_cv, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)

    st.image(cartoon, caption="Tor Cartoon Photo", use_container_width=True)
    st.download_button("Download Kor", cv2.imencode(".jpg", cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))[1].tobytes(), file_name="cartoon.jpg")
else:
    st.write("Upor e photo upload kor")
