import napari
import torch
import nibabel as nib
import numpy as np
import skimage.transform
from skimage import morphology
from skimage.measure import find_contours
from skimage.measure import approximate_polygon
from scipy.ndimage import gaussian_filter1d

def smooth_contour(contour, sigma=2):
    x, y = contour[:, 0], contour[:, 1]
    x_smooth = gaussian_filter1d(x, sigma)
    y_smooth = gaussian_filter1d(y, sigma)
    return np.stack([x_smooth, y_smooth], axis=1)


# Load .pt file
data_pt = torch.load('BraTS-GLI-00001-000-t2w_slice80.nii_0_output_ens.pt', map_location=torch.device('cpu'))
if isinstance(data_pt, torch.Tensor):
    array_pt = data_pt.cpu().numpy()
else:
    array_pt = np.array(data_pt)

# Remove singleton dimension if present
if array_pt.ndim == 3 and array_pt.shape[0] == 1:
    array_pt = array_pt[0]  # Now shape is (256, 256)

# Resize and normalize
print("array_pt shape/min/max:", array_pt.shape, array_pt.min(), array_pt.max())

array_pt = skimage.transform.resize(array_pt, (240, 240), preserve_range=True, anti_aliasing=True)
array_pt = (array_pt - array_pt.min()) / (array_pt.max() - array_pt.min())  # Normalize to [0,1]

print("array_pt shape/min/max:", array_pt.shape, array_pt.min(), array_pt.max())

# Threshold and denoise
binary_mask = (array_pt > 0.13).astype(np.uint8)
clean_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=200)

# Find contours
contours = find_contours(clean_mask, level=0.5)

# Simplify each contour
simplified_contours = []
for contour in contours:
    simplified = approximate_polygon(contour, tolerance=2.0) # Increase tolerance to reduce vertices more aggressively
    if len(simplified) >= 3:  # valid polygon
        simplified_contours.append(simplified)

smoothed_contours = [smooth_contour(c) for c in simplified_contours]

# Napari viewer
viewer = napari.Viewer()
viewer.add_image(array_pt, name='PT Data')

if contours:
    viewer.add_shapes(
        smoothed_contours,
        shape_type='polygon',
        edge_color='cyan',
        edge_width=2,
        name='Editable Contour'
    )
    viewer.add_shapes(
        simplified_contours,
        shape_type='polygon',
        edge_color='red',
        edge_width=2,
        name='Contour'
    )
else:
    print("No valid contour found.")

napari.run()
