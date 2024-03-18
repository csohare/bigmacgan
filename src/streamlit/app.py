from select import select
import streamlit as st
import os

st.set_page_config(layout="wide")
default_path = os.getcwd()
modelPath = os.path.join(default_path, "..", "models", "supervised")

st.markdown("<h1 style='text-align: center; color: white;'>BigMacGan Final Project</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap="large")
with col1:
    options = ["UNet", "FCNResnetTransfer", "SegmentationCNN", "DeepLabV3"]
    selected_option = st.selectbox("Select Model", options, )

dataPath = os.path.join(default_path, selected_option)

if selected_option == "UNet":
    model = "unet.py"
elif selected_option == "FCNResnetTransfer":
    model = "resnet_transfer.py"
elif selected_option == "SegmentationCNN":
    model = "segmentation_cnn.py"
elif selected_option == "DeepLabV3":
    model = "deeplabv3.py"



with col1:
    st.write("### Model Architecture")
    st.image(os.path.join(dataPath, 'arch.png'))
    with open(os.path.join(dataPath, 'check.ckpt'), "rb") as f:
        fileContents = f.read()
    st.download_button(
        label=f'Download {selected_option} Model Checkpoint',
        data = fileContents,
        file_name = f'{selected_option}.ckpt'
    )

with col2:
    st.write(f'### Example {selected_option} Model Outputs')
    for file in os.listdir(dataPath):
        if file.endswith(".png") and file != "arch.png":
            st.image(os.path.join(dataPath, file), use_column_width=True, width=500)

with col3:
    st.write(f'### {selected_option} Model Code')
    with open(os.path.join(modelPath, model), "r") as f:
        code = f.read()
    st.code(code)


# with open(checkpointPath, "rb") as f:
#     fileContents = f.read()

# st.download_button(
#     label= 'Download UNet Params',
#     data= fileContents,
#     file_name= 'UNet.ckpt'
# )






