import numpy as np
import torch
import skimage.transform
from skimage import morphology
from skimage.measure import find_contours
from scipy.interpolate import BSpline
from scipy.ndimage import center_of_mass
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes


# Load .pt mask
data_pt = torch.load("BraTS-GLI-00001-000-t2w_slice80.nii_0_output_ens.pt", map_location="cpu")
array_pt = data_pt.numpy() if isinstance(data_pt, torch.Tensor) else np.array(data_pt)
if array_pt.ndim == 3 and array_pt.shape[0] == 1:
    array_pt = array_pt[0]
array_pt = skimage.transform.resize(array_pt, (240, 240), preserve_range=True, anti_aliasing=True)
array_pt = (array_pt - array_pt.min()) / (array_pt.max() - array_pt.min())

# Clean mask
binary = array_pt > 0.13
mask = morphology.remove_small_objects(binary, min_size=200)

# Center of mass
cy, cx = center_of_mass(mask)

# Extract polar contour
contour = find_contours(mask, 0.5)[0]
angles = np.arctan2(contour[:, 0] - cy, contour[:, 1] - cx)
radii = np.sqrt((contour[:, 0] - cy) ** 2 + (contour[:, 1] - cx) ** 2)
sorted_idx = np.argsort(angles)
angles_sorted = angles[sorted_idx]
radii_sorted = radii[sorted_idx]

# Fit periodic B-spline
num_knots = 80
theta_knots = np.linspace(-np.pi, np.pi, num_knots, endpoint=False)
r_samples = np.interp(theta_knots, angles_sorted, radii_sorted)
r_ext = np.concatenate([r_samples[-3:], r_samples, r_samples[:3]])
t = np.concatenate([
    theta_knots[-3:] - 2*np.pi,
    theta_knots,
    theta_knots[:3] + 2*np.pi
])
k = 3  # cubic
spl = BSpline(t, r_ext, k, extrapolate='periodic')

def circular_second_diff_penalty(c):
    return np.sum((np.roll(c, -1) - 2*c + np.roll(c, 1))**2)

# Localised Yezzi energy (equation 3 in the paper):
# g(θ) = [(I(ψ(θ), θ) - u_in)^2 / A_in] - [(I(ψ(θ), θ) - u_out)^2 / A_out]
# We simplify by minimizing:
# E(c) = Σ_in (I - u)^2 + Σ_out (I - v)^2

def yezzi_energy_smooth(c_opt, lam=0.005):
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
    data_energy = ((inside - u)**2).sum() + ((outside - v)**2).sum()
    smooth_penalty = lam * circular_second_diff_penalty(c_opt)
    return data_energy + smooth_penalty


res = minimize(yezzi_energy_smooth, r_samples, method='L-BFGS-B')
c_final = np.concatenate([res.x[-3:], res.x, res.x[:3]])
spl_opt = BSpline(t, c_final, k, extrapolate='periodic')

# Visualize
theta_fine = np.linspace(-np.pi, np.pi, 500, endpoint=False)
r_fine = spl_opt(theta_fine)
x_f = cx + r_fine * np.cos(theta_fine)
y_f = cy + r_fine * np.sin(theta_fine)

plt.figure(figsize=(5, 5))
plt.imshow(array_pt, cmap="gray")
plt.plot(contour[:, 1], contour[:, 0], "r--", label="Raw Contour")
plt.plot(x_f, y_f, "c-", linewidth=2, label="BEAS")
plt.legend()
plt.title("BEAS Optimized Contour")
plt.axis("off")
plt.show()
