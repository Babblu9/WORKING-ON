import subprocess
import shutil
import time
import streamlit as st
from PIL import Image
import os

# Function to save uploaded files
def save_uploaded_file(uploaded_file, path):
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# Function to run GMM model
def run_gmm_test():
    gmm_test = ("python test.py --name GMM --stage GMM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint "
                "checkpoints/GMM/gmm_final.pth")
    subprocess.call(gmm_test, shell=True)
    warp_cloth = "result/GMM/test/warp-cloth"
    warp_mask = "result/GMM/test/warp-mask"
    shutil.copytree(warp_cloth, "data/test/warp-cloth", dirs_exist_ok=True)
    shutil.copytree(warp_mask, "data/test/warp-mask", dirs_exist_ok=True)
    return "result/GMM/test/warp-cloth/00001.png", "result/GMM/test/warp-mask/00001.png"

# Function to run TOM model
def run_tom_test():
    tom_test = ("python test.py --name TOM --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint "
                "checkpoints/TOM/tom_final.pth")
    subprocess.call(tom_test, shell=True)
    return "result/TOM/test/try-on/00001.png"

# Streamlit UI
st.title('Real-Time Machine Learning Model Deployment')

st.header('Upload Images for Real-Time Processing')

# Upload user photo
uploaded_photo = st.file_uploader("Upload a photo of yourself", type=["png", "jpg", "jpeg"])
if uploaded_photo is not None:
    user_photo_path = save_uploaded_file(uploaded_photo, "data/test/person.png")
    st.image(Image.open(uploaded_photo), caption="Uploaded Photo", use_column_width=True)

# Upload cloth photo
uploaded_cloth = st.file_uploader("Upload a photo of the cloth", type=["png", "jpg", "jpeg"])
if uploaded_cloth is not None:
    cloth_photo_path = save_uploaded_file(uploaded_cloth, "data/test/cloth.png")
    st.image(Image.open(uploaded_cloth), caption="Uploaded Cloth", use_column_width=True)

st.header('GMM Model')
if st.button('Run GMM Test'):
    if uploaded_photo is None or uploaded_cloth is None:
        st.error("Please upload both a photo of yourself and a cloth before running the test.")
    else:
        start_time = time.time()
        gmm_cloth_result, gmm_mask_result = run_gmm_test()
        st.success(f"GMM Test completed in {time.time() - start_time} seconds")
        st.image(Image.open(gmm_cloth_result), caption="Warped Cloth Result", use_column_width=True)
        st.image(Image.open(gmm_mask_result), caption="Warped Mask Result", use_column_width=True)

st.header('TOM Model')
if st.button('Run TOM Test'):
    if uploaded_photo is None or uploaded_cloth is None:
        st.error("Please upload both a photo of yourself and a cloth before running the test.")
    else:
        start_time = time.time()
        tom_result = run_tom_test()
        st.success(f"TOM Test completed in {time.time() - start_time} seconds")
        st.image(Image.open(tom_result), caption="Try-On Result", use_column_width=True)

st.write("All processes finished.")
