# run_full_pipeline.py
import os
import numpy as np
import torch as th
import nibabel as nib
import skimage.transform
import shutil
import datetime
import napari
import sys
from qtpy.QtWidgets import QPushButton, QMessageBox, QLabel
from qtpy.QtCore import QThread, Signal, QObject
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.segmentation_sample import sample_once, create_argparser, create_model, load_data

NUM_CANDIDATES = 3
DATA_DIR = "./BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
OUTPUT_DIR = "interaction_phase2_output"
SELECTED_DIR = os.path.join(OUTPUT_DIR, "user_selected")

selected_mask = None
selected_mask_path = None
original_slice = None
slice_idx = 80  # For demo purpose
output_paths = []

def load_pt_as_np(path):
    mask = th.load(path, map_location=th.device('cpu'))
    mask = mask.numpy() if isinstance(mask, th.Tensor) else np.array(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask = skimage.transform.resize(mask, (240, 240), preserve_range=True, anti_aliasing=True)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    # if isinstance(mask, th.Tensor):
    #     mask = mask.cpu().numpy()
    # mask = skimage.transform.resize(mask, (240, 240), preserve_range=True, anti_aliasing=True)
    # mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask

def parse_output_path(output_path):
    filename = os.path.basename(output_path)
    base = filename.split('_output')[0]
    filename_part, slice_part = base.split('_slice')
    slice_id = int(slice_part.split('.')[0])
    folder = '-'.join(filename_part.split('-')[:4])
    return folder, filename_part, slice_id

def load_candidate_masks_with_diffusion():
    global selected_mask, selected_mask_path, original_slice, slice_idx
    args = create_argparser().parse_args()
    # === Define all args manually ===
    args.data_name = "BRATS"
    args.data_dir = "C:/Users/26364/Documents/IC/IC_Y4/FYP/MedSegDiff/BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    args.out_dir = "C:/Users/26364/Documents/IC/IC_Y4/FYP/MedSegDiff/interaction_phase2_output"
    args.model_path = "C:/Users/26364/Documents/IC/IC_Y4/FYP/MedSegDiff/output/emasavedmodel_0.9999_085000.pt"
    args.image_size = 256
    args.num_channels = 128
    args.class_cond = False
    args.num_res_blocks = 2
    args.num_heads = 1
    args.learn_sigma = True
    args.use_scale_shift_norm = False
    args.attention_resolutions = "16"
    args.diffusion_steps = 50
    args.noise_schedule = "linear"
    args.rescale_learned_sigmas = False
    args.rescale_timesteps = False
    args.dpm_solver = True
    args.num_ensemble = 5

    datal = load_data(args)
    model, diffusion = create_model(args)

    masks = []
    global output_paths 
    output_paths = []
    for i in range(NUM_CANDIDATES):
        output_path = sample_once(slice_idx, i, args, datal, model, diffusion)
        mask = load_pt_as_np(output_path)
        masks.append(mask)
        output_paths.append(output_path)

    # selected_mask_path = output_paths[0]

    folder, filename, _ = parse_output_path(output_paths[0])
    nii = nib.load(f"{DATA_DIR}/{folder}/{filename}.nii.gz")
    original = nii.get_fdata()
    original = np.moveaxis(original, -1, 0)
    original_slice = original[slice_idx]
    
    global viewer 
    viewer= napari.Viewer()

    viewer.add_image(original_slice, name="Original Slice")

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    layer_names = []

    for idx, mask in enumerate(masks):
        layer = viewer.add_image(
            mask,
            name=f"Candidate Mask {idx}",
            opacity=0.15,
            colormap=colors[idx % len(colors)],
            blending="additive"
        )
        layer_names.append(layer.name)

def save_selected_mask():
    global selected_mask, selected_mask_path
    selected_layer = viewer.layers.selection.active
    if selected_layer is None:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Please select a mask layer before clicking Save.")
        msg.setWindowTitle("No Layer Selected")
        msg.exec_()
        return

    idx = int(selected_layer.name.split()[-1])
    selected_mask_path = output_paths[idx]
    chosen_output_dir = os.path.join('interaction_phase2_output', 'user_selected')
    print(f"Saving selected mask to {chosen_output_dir}")
    os.makedirs(chosen_output_dir, exist_ok=True)
    shutil.copy(
        os.path.join('interaction_phase2_output', os.path.basename(selected_mask_path)),
        os.path.join(chosen_output_dir, os.path.basename(selected_mask_path))
    )
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(f"Chosen mask saved to {chosen_output_dir}")
    msg.setWindowTitle("Mask Saved")
    msg.exec_()

    #selected_mask_path = f"{OUTPUT_DIR}/BraTS-GLI-00001-000-t2w_slice{slice_idx}_{idx}_output_ens.pt"
    selected_mask = load_pt_as_np(selected_mask_path)
    print(f"Selected mask loaded from {selected_mask_path}")

    viewer.window.add_dock_widget(QPushButton("Proceed to BEAS", clicked=run_beas), area="right")
    viewer.window.add_dock_widget(QPushButton("Proceed to GraphCut", clicked=run_graphcut), area="right")

def run_beas():
    from beas import run_beas_refinement
    if selected_mask is None:
        print("No mask selected.")
        return

    run_beas_refinement(viewer, original_slice, selected_mask)

def run_graphcut():
    from graphcut import run_graphcut_ui
    if selected_mask is None:
        print("No mask selected.")
        return

    run_graphcut_ui(viewer, original_slice, selected_mask)

def main():
    load_candidate_masks_with_diffusion()
    proceed_button = QPushButton("Select Mask and Proceed")
    proceed_button.clicked.connect(save_selected_mask)
    viewer.window.add_dock_widget(proceed_button, area="right")

    instruction = QLabel("Select a mask and proceed to BEAS or GraphCut")
    instruction.setWordWrap(True)
    viewer.window.add_dock_widget(instruction, area="right")
    napari.run()

if __name__ == "__main__":
    main()