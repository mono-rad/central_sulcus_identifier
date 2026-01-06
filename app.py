import streamlit as st
import os
import tempfile
import zipfile
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from skimage import measure
from torch.utils.data import DataLoader, TensorDataset
import dicom2nifti
import glob

# --- Configuration ---
MODEL_DIR = "models"
DEMO_DIR = "demo_data"
# Free tier clouds (Streamlit Cloud) often lack GPU or have limited memory.
# Explicitly forcing CPU ensures stability during inference.
DEVICE = torch.device('cpu') 
BATCH_SIZE = 8
BEST_FOLD_INDEX = 1  # Best performing fold based on CV score (Dice: 0.9185)

# --- Utility Functions ---
def apply_windowing(image, level, width):
    """
    Standard DICOM windowing.
    Clip values to [level - width/2, level + width/2] and normalize to 0-1.
    """
    lower = level - (width / 2)
    upper = level + (width / 2)
    img_windowed = np.clip(image, lower, upper)
    
    # Avoid division by zero if window width is 0 (though unlikely in valid MRI)
    if upper - lower != 0:
        return (img_windowed - lower) / (upper - lower)
    else:
        return img_windowed - lower

def dicom_to_nifti(zip_file):
    """
    Handles ZIP upload containing DICOM series.
    Streamlit doesn't expose a persistent path, so we use a temp dir.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        dicom_dir = os.path.join(tmpdirname, "dicom")
        os.makedirs(dicom_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dicom_dir)
        
        output_dir = os.path.join(tmpdirname, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # dicom2nifti creates .nii.gz files in the output directory
            dicom2nifti.convert_directory(dicom_dir, output_dir, compression=True, reorient=True)
        except Exception as e:
            # Conversion often fails with non-axial or inconsistent DICOMs
            return None
        
        nii_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
        if not nii_files: return None
        
        # Load immediately to keep data in memory before temp dir is cleaned up
        temp_nii = nib.load(os.path.join(output_dir, nii_files[0]))
        new_nii = nib.Nifti1Image(temp_nii.get_fdata(), temp_nii.affine, temp_nii.header)
        return new_nii

# --- Post-processing Logic ---
def apply_post_processing(prob_map, threshold, min_area, center_mask_ratio):
    """
    Refines the raw probability map to remove False Positives.
    Pipeline: Threshold -> Center Mask -> Split Hemispheres -> Keep Largest Component
    """
    # 1. Binary Thresholding
    if prob_map.max() < threshold: return np.zeros_like(prob_map, dtype=np.uint8)
    mask = (prob_map > threshold).astype(np.uint8)
    
    h, w = mask.shape
    mid_w = w // 2
    
    # 2. Center Masking (Optional)
    # Removes noise in deep brain structures (e.g., ventricles, basal ganglia) 
    # where the central sulcus never exists.
    if center_mask_ratio > 0:
        box_h = int(h * center_mask_ratio) // 2
        box_w = int(w * center_mask_ratio) // 2
        # Define the exclusion box in the center
        danger_top, danger_left = h // 2 - box_h, w // 2 - box_w
        danger_bottom, danger_right = h // 2 + box_h, w // 2 + box_w
        
        labels = measure.label(mask, connectivity=2) # 8-connectivity (diagonal included)
        props = measure.regionprops(labels)
        for prop in props:
            min_r, min_c, max_r, max_c = prop.bbox
            # If a component is entirely within the center box, kill it.
            if (min_r >= danger_top and max_r <= danger_bottom and 
                min_c >= danger_left and max_c <= danger_right):
                mask[labels == prop.label] = 0

    # 3. Hemisphere Processing
    # Anatomical assumption: There is typically one continuous central sulcus per hemisphere.
    # We split the image and keep only the largest component on each side.
    left_mask = mask[:, :mid_w]; right_mask = mask[:, mid_w:]
    
    def filter_largest(submask):
        labels = measure.label(submask, connectivity=2)
        if labels.max() == 0: return submask
        props = measure.regionprops(labels)
        
        # Identify the largest component
        idx = np.argmax([p.area for p in props])
        
        # 4. Minimum Area Filter (Noise removal)
        if props[idx].area < min_area: return np.zeros_like(submask)
        return (labels == props[idx].label).astype(np.uint8)

    cleaned = np.zeros_like(mask)
    cleaned[:, :mid_w] = filter_largest(left_mask)
    cleaned[:, mid_w:] = filter_largest(right_mask)
    return cleaned

# --- Model Loader ---
@st.cache_resource
def load_models(mode):
    """
    Loads models into memory. Cached to prevent IO overhead on every interaction.
    """
    models = []
    # "Single" uses only the best fold for speed. "Ensemble" uses all 5 folds.
    target_folds = [BEST_FOLD_INDEX] if mode == "Single (Fast)" else range(5)
    
    for fold in target_folds:
        model_path = os.path.join(MODEL_DIR, f"best_model_fold{fold}.pth")
        if os.path.exists(model_path):
            model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models.append(model)
    return models

# --- Preprocessing ---
def preprocess_for_inference(volume_data):
    """
    Converts (H, W, D) numpy array to (D, C, H, W) torch tensor.
    Applies robust normalization (0.5-99.5 percentile) to handle MRI intensity outliers.
    """
    # Transpose to PyTorch format: (D, H, W)
    volume_tensor = np.transpose(volume_data, (2, 0, 1))
    tensor = torch.from_numpy(volume_tensor).float().unsqueeze(1) # Add channel dim: (D, 1, H, W)
    
    # Robust Min-Max Scaling
    # MRI pixel values are not standardized (unlike CT Hounsfield units).
    # We clip outliers (top/bottom 0.5%) before normalization to stabilize input.
    p05 = torch.quantile(tensor, 0.005)
    p995 = torch.quantile(tensor, 0.995)
    tensor = torch.clamp(tensor, p05, p995)
    
    if p995 - p05 != 0:
        tensor = (tensor - p05) / (p995 - p05)
    else:
        tensor = tensor * 0
    return tensor

# --- Main App Interface ---
st.set_page_config(page_title="Central Sulcus AI", layout="wide")

# CSS to fix sidebar width for better readability on mobile/tablets
st.markdown("""
    <style>
    [data-testid="stSidebar"] { min-width: 250px; max-width: 350px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 中心溝同定サポートAI")
# Disclaimer is crucial for avoiding medical liability
st.error("⚠️ **For Educational/Research Use Only (Not for Clinical Diagnosis)**\n\n本ツールは教育・研究目的であり、臨床診断には使用できません。")

# --- Sidebar: Controls ---
st.sidebar.header("🔄 Orientation")
col_rot1, col_rot2 = st.sidebar.columns(2)

# Rotation logic: Updates state and clears previous prediction cache
if col_rot1.button("↺ 左回転"):
    if st.session_state.get('rot_k') is not None:
        st.session_state['rot_k'] = (st.session_state['rot_k'] + 1) % 4
        st.session_state['probs_data'] = None
        st.rerun()
if col_rot2.button("↻ 右回転"):
    if st.session_state.get('rot_k') is not None:
        st.session_state['rot_k'] = (st.session_state['rot_k'] - 1) % 4
        st.session_state['probs_data'] = None
        st.rerun()

st.sidebar.header("🔍 Image Settings")
wl_slider = st.sidebar.empty() # Placeholders to be updated after data load
ww_slider = st.sidebar.empty()

st.sidebar.header("⚙️ Inference Settings")
inference_mode = st.sidebar.radio("Mode", ("Single (Fast)", "Ensemble (High Accuracy)"), index=0)

st.sidebar.header("🛠 Post-processing")
with st.sidebar.expander("ℹ️ パラメータの解説 (Help)"):
    st.markdown("""
    **閾値 (Threshold)**
    AIの確信度です。値を**下げる**と薄いシグナルも拾いますがノイズが増えます。**上げる**と判定が厳しくなります。
    
    **最小面積 (Min Area)**
    ゴミ取りフィルタです。指定したピクセル数以下の**孤立した小さな領域**をノイズとみなして消去します。
    
    **中央マスク率 (Center Mask)**
    解剖学的フィルタです。中心溝が存在しない**脳深部**を強制的にマスクし、四丘体槽などの誤検出を防ぎます。
    """)

threshold = st.sidebar.slider("閾値", 0.0, 1.0, 0.7)
min_area = st.sidebar.slider("最小面積", 0, 200, 30)
center_mask = st.sidebar.slider("中央マスク率", 0.0, 0.5, 0.25)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Data Source**\n\n"
    "Dataset: RSNA Intracranial Aneurysm Detection\n"
    "Authors: Jeff Rudie, et al.\n"
    "Source: [Kaggle Competition](https://kaggle.com/competitions/rsna-intracranial-aneurysm-detection)\n\n"
    "**Demo Data Source**\n"
    "IXI Dataset ([brain-development.org](https://brain-development.org/ixi-dataset/))\n"
    "License: CC BY-SA 3.0\n"
    "*Note: Selected files have been renamed for demonstration purposes.*"
)

# --- Session State Management ---
# Initialize session variables to persist data across Streamlit reruns
if 'volume_raw' not in st.session_state: st.session_state['volume_raw'] = None   
if 'rot_k' not in st.session_state: st.session_state['rot_k'] = 1
if 'probs_data' not in st.session_state: st.session_state['probs_data'] = None
if 'wl_ww_init' not in st.session_state: st.session_state['wl_ww_init'] = None

# --- File Loading ---
st.write("### 📂 Upload MRI Data")
col_up1, col_up2 = st.columns([2, 1])

with col_up1:
    uploaded_file = st.file_uploader("NIfTI / DICOM-ZIP", type=['nii', 'nii.gz', 'zip'])

# Logic for loading Pre-packaged Demo Data
with col_up2:
    st.write("No Data?")
    search_pattern = os.path.join(DEMO_DIR, "**", "*.nii*")
    demo_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if demo_files:
        demo_options = {os.path.basename(f): f for f in demo_files}
        selected_demo_name = st.selectbox("Select Demo Case", list(demo_options.keys()))
        demo_path = demo_options[selected_demo_name]
        
        if st.button("📂 Load Demo"):
            # Reset state to avoid mixing data from previous uploads
            st.session_state['volume_raw'] = None
            st.session_state['rot_k'] = 1 # Reset to default orientation (Nose Up)
            st.session_state['probs_data'] = None
            st.session_state['wl_ww_init'] = None
            st.session_state['last_uploaded'] = f"DEMO_{selected_demo_name}"
            
            nii = nib.as_closest_canonical(nib.load(demo_path))
            raw_vol = nii.get_fdata()
            
            # Reorder axes if necessary. Here we flip Z to match Head First orientation.
            # (Based on empirical observation of data format)
            raw_vol = raw_vol[:, :, ::-1]
            st.session_state['volume_raw'] = raw_vol
            
            # Auto-calculate Window Level/Width based on percentiles
            p1, p99 = np.percentile(raw_vol, 1), np.percentile(raw_vol, 99)
            st.session_state['wl_ww_init'] = ((p99+p1)/2, p99-p1, float(np.min(raw_vol)), float(np.max(raw_vol)))
            st.success(f"{selected_demo_name} を読み込みました")
            st.rerun()
    else:
        st.button("📂 Load Demo", disabled=True, help="デモデータが見つかりません。")
        if os.path.exists(DEMO_DIR):
            st.caption(f"※ No NIfTI files found in `{DEMO_DIR}`")
        else:
            st.caption(f"※ Folder `{DEMO_DIR}` not found.")

# Logic for Handling User Uploads
if uploaded_file:
    # Check if a new file is uploaded to trigger reload
    if st.session_state.get('last_uploaded') != uploaded_file.name:
        st.session_state['volume_raw'] = None
        st.session_state['rot_k'] = 1
        st.session_state['probs_data'] = None
        st.session_state['wl_ww_init'] = None
        st.session_state['last_uploaded'] = uploaded_file.name
        
        with st.spinner('Loading...'):
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext == 'zip':
                nii = dicom_to_nifti(uploaded_file)
                if nii is None: st.error("DICOM変換失敗"); st.stop()
            else:
                # Need to save to temp file because nibabel requires a file path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
                    tmp.write(uploaded_file.read()); tmp_path = tmp.name
                nii = nib.as_closest_canonical(nib.load(tmp_path))
                os.remove(tmp_path)
            
            raw_vol = nii.get_fdata()
            raw_vol = raw_vol[:, :, ::-1] # Z-flip for Head First
            st.session_state['volume_raw'] = raw_vol
            
            p1, p99 = np.percentile(raw_vol, 1), np.percentile(raw_vol, 99)
            st.session_state['wl_ww_init'] = ((p99+p1)/2, p99-p1, float(np.min(raw_vol)), float(np.max(raw_vol)))

# --- Viewer & Inference ---
if st.session_state['volume_raw'] is not None:
    # Apply rotation (User interactive)
    volume_display = np.rot90(st.session_state['volume_raw'], k=st.session_state['rot_k']).copy()
    max_slice = volume_display.shape[2] - 1
    init_wl, init_ww, min_val, max_val = st.session_state['wl_ww_init']
    
    # UI Sliders
    wl = wl_slider.slider("Window Level", float(min_val), float(max_val), float(init_wl))
    ww = ww_slider.slider("Window Width", 1.0, float(max_val - min_val)*1.5, float(init_ww))
    
    st.divider()
    slice_idx = st.slider("Slice Explorer (0 = Head Side)", 0, max_slice, 0)
    
    col1, col2 = st.columns(2)
    
    # Column 1: Original Image
    with col1:
        st.subheader("👁️ Original Image")
        current_img_disp = apply_windowing(volume_display[:, :, slice_idx], wl, ww)
        
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.imshow(current_img_disp, cmap='gray', vmin=0, vmax=1)
        ax1.axis('off')
        st.pyplot(fig1, use_container_width=False)
        st.caption(f"Rotate: {st.session_state['rot_k'] * 90}°")
        # Note for medical professionals regarding orientation
        st.caption("※ Display Orientation: Neurological View (Left is Left)") 
        
        # Inference Trigger
        if st.session_state['probs_data'] is None:
            if st.button("🚀 Analyze (Nose Up)"):
                with st.spinner(f"AI ({inference_mode}) is thinking..."):
                    # Preprocess whole volume
                    input_tensor = preprocess_for_inference(volume_display)
                    models = load_models(inference_mode)
                    if not models: st.error("モデルなし"); st.stop()
                    
                    dataset = TensorDataset(input_tensor)
                    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
                    prob_slices = []
                    
                    # Inference Loop (Batched for memory efficiency)
                    bar = st.progress(0)
                    for i, batch in enumerate(loader):
                        imgs = batch[0]
                        ens_prob = torch.zeros_like(imgs)
                        # Average predictions across models (Ensemble)
                        for m in models:
                            with torch.no_grad(): ens_prob += torch.sigmoid(m(imgs))
                        ens_prob /= len(models)
                        prob_slices.append(ens_prob.squeeze(1).numpy())
                        bar.progress((i+1)/len(loader))
                    bar.empty()
                    
                    # Reconstruct 3D volume from 2D slices
                    probs_vol = np.concatenate(prob_slices, axis=0)
                    probs_vol = np.transpose(probs_vol, (1, 2, 0)) # (D, H, W) -> (H, W, D)
                    
                    st.session_state['probs_data'] = probs_vol
                    st.rerun()

    # Column 2: Prediction Result
    with col2:
        if st.session_state['probs_data'] is not None:
            st.subheader("🤖 AI Prediction")
            probs = st.session_state['probs_data']
            current_prob = probs[:, :, slice_idx]
            
            # Apply real-time post-processing based on slider values
            final_mask = apply_post_processing(current_prob, threshold, min_area, center_mask)
            
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.imshow(current_img_disp, cmap='gray', vmin=0, vmax=1)
            
            # Overlay heatmap only if mask exists
            if np.max(final_mask) > 0:
                ax2.imshow(final_mask, cmap='jet', alpha=0.5)
            ax2.axis('off')
            st.pyplot(fig2, use_container_width=False)
            
            if st.button("🔄 Reset Analysis"):
                st.session_state['probs_data'] = None
                st.rerun()
        else:
            st.empty()