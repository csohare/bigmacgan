from select import select
import streamlit as st
import os
def save_uploaded_folder(uploaded_folder, save_path):
    os.makedirs(save_path, exist_ok=True)
    for root, dirs, files in os.walk(uploaded_folder):
        for file in files:
            st.write(file)


st.set_page_config(layout="wide")
default_path = os.path.join(os.getcwd(), 'src', 'streamlit')
modelPath = os.path.join(os.getcwd(), 'src' , "models", "supervised")

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

    folder_upload_button = """
    <div style="display: flex; justify-content: center;">
    <label class="upload-btn">
        <input type="file" id="fileElem" webkitdirectory mozdirectory msdirectory odirectory directory multiple />
        Upload Folder
    </label>
    </div>

    <script>
    const fileElem = document.getElementById("fileElem");
    const uploadBtn = document.querySelector(".upload-btn");

    uploadBtn.addEventListener("click", (e) => {
        fileElem.click();
    });
    </script>
    """
    st.markdown(folder_upload_button, unsafe_allow_html=True)
    if st.button("Save Uploaded Folder"):
        st.write(st.session_state)
        if "fileElem" not in st.session_state:
            st.error("No folder uploaded.")
        else:
            uploaded_folder = st.session_state["fileElem"]
            save_uploaded_folder(uploaded_folder, default_path)
            st.success("Folder saved successfully.")


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






