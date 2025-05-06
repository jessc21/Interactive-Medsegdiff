import napari
import torch
import nibabel as nib
import numpy as np

# For .pt file (assuming tensor or array saved)
data_pt = torch.load('BraTS-GLI-00474-001-t2w_slice131.nii_1_output_ens.pt')
if isinstance(data_pt, torch.Tensor):
    array_pt = data_pt.cpu().numpy()
else:
    array_pt = np.array(data_pt)

# For .nii.gz file
nii = nib.load('BraTS-GLI-01666-000-seg.nii.gz')
array_nii = nii.get_fdata()

# Launch napari viewer
viewer = napari.Viewer()
viewer.add_image(array_pt, name='PT Data')
viewer.add_image(array_nii, name='NIfTI Data')
