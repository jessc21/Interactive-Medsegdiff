import numpy as np
import maxflow
import torch
from qtpy.QtWidgets import QPushButton
from skimage import morphology
import skimage.transform
import nibabel as nib

def run_graphcut_ui(viewer, original_slice, selected_mask):
    """
    Launch GraphCut UI with a button to save annotations and apply refinement.

    Parameters:
    - viewer: napari.Viewer instance
    - original_slice: 2D image array (normalized)
    - selected_mask: binary segmentation mask (from CNN or BEAS)
    """
    data_pt = torch.load("BraTS-GLI-00001-000-t2w_slice80.nii_0_output_ens.pt", map_location="cpu")
    array_pt = data_pt.numpy() if isinstance(data_pt, torch.Tensor) else np.array(data_pt)

    #array_pt = selected_mask.numpy() if isinstance(selected_mask, torch.Tensor) else np.array(selected_mask)
    print(array_pt.shape, array_pt.min(), array_pt.max())
    if array_pt.ndim == 3 and array_pt.shape[0] == 1:
        array_pt = array_pt[0]
    array_pt = skimage.transform.resize(array_pt, (240, 240), preserve_range=True, anti_aliasing=True)
    array_pt = (array_pt - array_pt.min()) / (array_pt.max() - array_pt.min())
    print("array_pt shape/min/max:", array_pt.shape, array_pt.min(), array_pt.max())
    # Clean initial mask
    initial_mask = morphology.remove_small_objects(array_pt > 0.15, min_size=200).astype(np.uint8)

    # Display layers
    #viewer.add_image(original_slice, name='Original Slice', colormap='gray', blending='additive', opacity=0.5)
    for layer in list(viewer.layers):
        if layer.name.startswith("Candidate mask"):
            viewer.layers.remove(layer)
    if "Chosen Mask" in viewer.layers:
        viewer.layers.remove(viewer.layers["Chosen Mask"])

    viewer.add_image(array_pt, name='Chosen Mask', colormap='gray', opacity=0.7)
    viewer.add_labels(initial_mask, name='Initial Graphcut Mask')
    viewer.add_labels(np.zeros_like(initial_mask, dtype=np.uint8), name='User Labels')  # 1: FG, 2: BG

    # ---------------- GraphCut logic ----------------
    def run_graph_cut(image, user_labels):
        print("image shape:", image.shape)
        h, w = image.shape
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((h, w))

        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])

        # Smoothness term
        weights = np.exp(-((image - np.mean(image)) ** 2) * 50)
        g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=True)

        # Terminal links
        for y in range(h):
            for x in range(w):
                if user_labels[y, x] == 1:
                    g.add_tedge(nodeids[y, x], 1e10, 0)
                elif user_labels[y, x] == 2:
                    g.add_tedge(nodeids[y, x], 0, 1e10)

        g.maxflow()
        refined = g.get_grid_segments(nodeids)
        return (~refined).astype(np.uint8)

    # ---------------- Button Action ----------------
    def on_save_clicked():
        user_labels = viewer.layers["User Labels"].data
        refined = run_graph_cut(array_pt, user_labels)

        init_fg = (initial_mask > 0)
        refined_fg = (refined == 1)
        refined_bg = (user_labels == 2)

        # Merge with initial
        final_mask = (init_fg | refined_fg)  # optionally: & (~refined_bg)

        viewer.add_labels(final_mask, name="Refined Mask", opacity=0.6)

        # Optional: Save to NIfTI
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(final_mask.astype(np.uint8), affine), "graphcut_output_seg.nii.gz")

    # ---------------- Add Button ----------------
    save_btn = QPushButton("Save Annotations and Run GraphCut")
    save_btn.clicked.connect(on_save_clicked)
    viewer.window.add_dock_widget(save_btn, area='right')
