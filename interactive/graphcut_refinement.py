import numpy as np
import torch
import skimage.transform
from skimage import morphology
import napari
from qtpy.QtWidgets import QPushButton
import maxflow
from scipy.stats import norm


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
# Graph Cut Refinement Function (updated)
# ------------------
def run_graph_cut(image, user_labels):
    h, w = image.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((h, w))

    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])
    # high beta Preserves object boundaries better, avoids over-segmentation
    beta = 50
    lam = 0.5

    # Estimate Gaussian distributions from seeds
    fg_vals = image[user_labels == 1]
    bg_vals = image[user_labels == 2]

    if len(fg_vals) < 5 or len(bg_vals) < 5:
        print("Insufficient foreground/background labels for probability modeling")
        return np.zeros_like(image, dtype=np.uint8)

    mu_fg, sigma_fg = np.mean(fg_vals), np.std(fg_vals) + 1e-6
    mu_bg, sigma_bg = np.mean(bg_vals), np.std(bg_vals) + 1e-6

    # Calculate R_p using negative log likelihood
    Rp_fg = -np.log(norm.pdf(image, mu_fg, sigma_fg) + 1e-8)
    Rp_bg = -np.log(norm.pdf(image, mu_bg, sigma_bg) + 1e-8)

    # Compute n-link weights
    diff = np.diff(image, axis=0, append=image[-1:, :])**2
    weights = np.exp(-beta * diff)
    g.add_grid_edges(nodeids, weights=weights, structure=structure, symmetric=True)

    # Compute K
    max_neighbors = np.max([np.sum(weights[i]) for i in range(weights.shape[0])])
    K = 1 + max_neighbors

    for y in range(h):
        for x in range(w):
            idx = nodeids[y, x]
            label = user_labels[y, x]

            if label == 1:  # Foreground seed
                g.add_tedge(idx, K, 0)
            elif label == 2:  # Background seed
                g.add_tedge(idx, 0, K)
            else:
                g.add_tedge(idx, lam * Rp_bg[y, x], lam * Rp_fg[y, x])

    g.maxflow()
    refined = g.get_grid_segments(nodeids)
    return (~refined).astype(np.uint8)


# ------------------
# Save and Run GraphCut Handler
# ------------------
def on_save_clicked():
    user_labels = viewer.layers["User Labels"].data
    refined = run_graph_cut(array_pt, user_labels)
    init_fg = (initial_mask > 0)
    refined_fg = (refined == 1)
    refined_bg = (user_labels == 2)

    final_mask = (init_fg | refined_fg) & (~refined_bg)
    viewer.add_labels(final_mask, name="Refined Mask", opacity=0.6)


# ------------------
# Launch Napari with Button
# ------------------
viewer = napari.Viewer()
viewer.add_image(array_pt, name='Image', colormap='gray')
viewer.add_labels(initial_mask, name='Initial Mask')
viewer.add_labels(np.zeros_like(initial_mask, dtype=np.uint8), name='User Labels')

save_btn = QPushButton("Save Annotations & Run GraphCut")
save_btn.clicked.connect(on_save_clicked)
viewer.window.add_dock_widget(save_btn, area='right')

napari.run()
