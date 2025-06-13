import numpy as np
import torch
import skimage.transform
from skimage import morphology
import napari
from napari.qt import thread_worker
from qtpy.QtWidgets import QPushButton
import maxflow
import nibabel as nib
import os

# Load the corresponding original T2w slice (adjust path accordingly)
nii = nib.load("../BraTS-GLI-00001-000-t2w.nii.gz")
original_volume = nii.get_fdata()
original_volume = np.moveaxis(original_volume, -1, 0)  # (H, W, D) → (D, H, W)
original_slice = skimage.transform.resize(original_volume[80], (240, 240), preserve_range=True, anti_aliasing=True)
original_slice = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min())

# ------------------
# Load probability map
# ------------------
data_pt = torch.load("../BraTS-GLI-00001-000-t2w_slice80.nii_0_output_ens.pt", map_location="cpu")
array_pt = data_pt.numpy() if isinstance(data_pt, torch.Tensor) else np.array(data_pt)
if array_pt.ndim == 3 and array_pt.shape[0] == 1:
    array_pt = array_pt[0]
array_pt = skimage.transform.resize(array_pt, (240, 240), preserve_range=True, anti_aliasing=True)
array_pt = (array_pt - array_pt.min()) / (array_pt.max() - array_pt.min())

# Initial mask (placeholder, can be from BEAS)
initial_mask = morphology.remove_small_objects(array_pt > 0.13, min_size=200).astype(np.uint8)

# ------------------
# Graph Cut Refinement Function
# ------------------
def run_graph_cut(image, user_labels):
    h, w = image.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((h, w))

    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])

    # Similarity edge weights
    weights = np.exp(-((image - np.mean(image)) ** 2) * 50)
    g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=True)

    # Set strong terminal weights only for labeled pixels
    for y in range(h):
        for x in range(w):
            if user_labels[y, x] == 1:  # foreground
                g.add_tedge(nodeids[y, x], 1e10, 0)
            elif user_labels[y, x] == 2:  # background
                g.add_tedge(nodeids[y, x], 0, 1e10)

    g.maxflow()
    refined = g.get_grid_segments(nodeids)
    return (~refined).astype(np.uint8)


# ------------------
# Save and Run GraphCut Handler
# ------------------
def on_save_clicked():
    user_labels = viewer.layers["User Labels"].data
    refined = run_graph_cut(array_pt, user_labels)
    init_fg = (initial_mask > 0)           # original segmentation (binary)
    refined_fg = (refined == 1)            # user-labeled FG
    refined_bg = (user_labels == 2)        # user-labeled BG

    final_mask = (init_fg | refined_fg) #& (~refined_bg)

    viewer.add_labels(final_mask, name="Refined Mask", opacity=0.6)

# ------------------
# Launch Napari with Button
# ------------------
viewer = napari.Viewer()
viewer.add_image(original_slice, name='Original Slice', colormap='gray', blending='additive', opacity=0.5)
viewer.add_image(array_pt, name='CNN Mask', colormap='gray', blending='additive', opacity=0.7)
viewer.add_labels(initial_mask, name='Initial Graphcut Mask')
viewer.add_labels(np.zeros_like(initial_mask, dtype=np.uint8), name='User Labels')  # 1: FG, 2: BG

# Add button to Napari
save_btn = QPushButton("Save Annotations & Run GraphCut")
save_btn.clicked.connect(on_save_clicked)
viewer.window.add_dock_widget(save_btn, area='right')

napari.run()