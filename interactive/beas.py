# test.py — BEAS module version
import numpy as np
import torch
import skimage.transform
from skimage import morphology
from skimage.measure import find_contours
from scipy.interpolate import BSpline
from scipy.ndimage import center_of_mass, binary_fill_holes
from scipy.optimize import minimize
from qtpy.QtWidgets import QPushButton
import nibabel as nib
import napari

def run_beas_refinement(viewer, image, mask_input):
    data_pt = torch.load("BraTS-GLI-00001-000-t2w_slice80.nii_0_output_ens.pt", map_location="cpu")
    array_pt = data_pt.numpy() if isinstance(data_pt, torch.Tensor) else np.array(data_pt)
    if array_pt.ndim == 3 and array_pt.shape[0] == 1:
        array_pt = array_pt[0]
    array_pt = skimage.transform.resize(array_pt, (240, 240), preserve_range=True, anti_aliasing=True)
    array_pt = (array_pt - array_pt.min()) / (array_pt.max() - array_pt.min())

    # array_pt = mask_input.numpy() if isinstance(mask_input, torch.Tensor) else np.array(mask_input)
    # if array_pt.ndim == 3 and array_pt.shape[0] == 1:
    #     array_pt = array_pt[0]
    # array_pt = skimage.transform.resize(array_pt, (240, 240), preserve_range=True, anti_aliasing=True)
    # array_pt = (array_pt - array_pt.min()) / (array_pt.max() - array_pt.min())

    # Normalize image
    image = skimage.transform.resize(image, (240, 240), preserve_range=True, anti_aliasing=True)
    image = (image - image.min()) / (image.max() - image.min())

    # Threshold and denoise
    binary = array_pt > 0.13
    mask = morphology.remove_small_objects(binary, min_size=200)

    # Get center of mass
    cy, cx = center_of_mass(mask)

    # Get raw contour
    contour = find_contours(mask, 0.5)[0]
    angles = np.arctan2(contour[:, 0] - cy, contour[:, 1] - cx)
    radii = np.sqrt((contour[:, 0] - cy) ** 2 + (contour[:, 1] - cx) ** 2)
    sorted_idx = np.argsort(angles)
    angles_sorted = angles[sorted_idx]
    radii_sorted = radii[sorted_idx]

    # B-spline setup
    num_knots = 50
    theta_knots = np.linspace(-np.pi, np.pi, num_knots, endpoint=False)
    r_samples = np.interp(theta_knots, angles_sorted, radii_sorted)
    r_ext = np.concatenate([r_samples[-3:], r_samples, r_samples[:3]])
    t = np.concatenate([theta_knots[-3:] - 2 * np.pi, theta_knots, theta_knots[:3] + 2 * np.pi])
    k = 3

    def circular_second_diff_penalty(c):
        return np.sum((np.roll(c, -1) - 2 * c + np.roll(c, 1)) ** 2)

    def yezzi_energy_smooth(c_opt, lam=1e-5):
        c_full = np.concatenate([c_opt[-3:], c_opt, c_opt[:3]])
        spline = BSpline(t, c_full, k, extrapolate='periodic')
        theta = np.linspace(-np.pi, np.pi, 240)
        r = spline(theta)
        x = np.clip(cx + r * np.cos(theta), 0, 239).astype(int)
        y = np.clip(cy + r * np.sin(theta), 0, 239).astype(int)

        mask = np.zeros_like(array_pt, dtype=np.uint8)
        mask[y, x] = 1
        mask = binary_fill_holes(mask).astype(np.uint8)

        inside = array_pt[mask == 1]
        outside = array_pt[mask == 0]
        if len(inside) == 0 or len(outside) == 0:
            return np.inf

        u, v = inside.mean(), outside.mean()
        data_energy = ((inside - u) ** 2).sum() + ((outside - v) ** 2).sum()
        smooth_penalty = lam * circular_second_diff_penalty(c_opt)
        return data_energy + smooth_penalty

    #viewer.add_image(image, name='Original Slice', colormap='gray', blending='additive', opacity=0.5)
    for layer in list(viewer.layers):
        if layer.name.startswith("Candidate mask"):
            viewer.layers.remove(layer)
    if "Chosen Mask" in viewer.layers:
        viewer.layers.remove(viewer.layers["Chosen Mask"])
    viewer.add_image(array_pt, name='Chosen Mask', colormap='gray', opacity=0.7)

    # Editable contour
    theta_edit = theta_knots
    x_edit = cx + r_samples * np.cos(theta_edit)
    y_edit = cy + r_samples * np.sin(theta_edit)
    editable_shape = np.stack([y_edit, x_edit], axis=1)
    shape_layer = viewer.add_shapes([editable_shape], shape_type='polygon', edge_color='yellow', name="Edit Knots")

    def on_run_optimization():
        user_shape = shape_layer.data[0]
        dx = user_shape[:, 1] - cx
        dy = user_shape[:, 0] - cy
        r_user = np.sqrt(dx ** 2 + dy ** 2)

        res = minimize(yezzi_energy_smooth, r_user, method='L-BFGS-B')
        c_final = np.concatenate([res.x[-3:], res.x, res.x[:3]])
        spl_final = BSpline(t, c_final, k, extrapolate='periodic')

        theta_fine = np.linspace(-np.pi, np.pi, 500, endpoint=False)
        r_fine = spl_final(theta_fine)
        x_f = cx + r_fine * np.cos(theta_fine)
        y_f = cy + r_fine * np.sin(theta_fine)
        final_contour = np.stack([y_f, x_f], axis=1)

        if "BEASContour" in viewer.layers:
            viewer.layers["BEASContour"].data = [final_contour]
        else:
            viewer.add_shapes([final_contour], shape_type='polygon', edge_color='cyan', name="BEASContour")
        
        # Convert polar contour to binary mask
        mask = np.zeros(array_pt.shape, dtype=np.uint8)
        rr = np.clip(y_f, 0, array_pt.shape[0] - 1).astype(int)
        cc = np.clip(x_f, 0, array_pt.shape[1] - 1).astype(int)
        mask[rr, cc] = 1
        filled_mask = binary_fill_holes(mask).astype(np.uint8)

        # Save to NIfTI format
        affine = np.eye(4)  # Or replace with affine from original NIfTI image
        seg_nifti = nib.Nifti1Image(filled_mask, affine=affine)
        nib.save(seg_nifti, "BraTS-GLI-00001-000-t2w_slice80-seg.nii.gz")

    button = QPushButton("Reoptimize BEAS")
    button.clicked.connect(on_run_optimization)
    viewer.window.add_dock_widget(button, area='right')

    napari.run()
