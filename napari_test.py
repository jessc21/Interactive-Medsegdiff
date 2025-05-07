import napari
import torch
import nibabel as nib
import numpy as np
import skimage.transform

# For .pt file (assuming tensor or array saved)
data_pt = torch.load('BraTS-GLI-00474-001-t2w_slice131.nii_1_output_ens.pt', map_location=torch.device('cpu'))
if isinstance(data_pt, torch.Tensor):
    array_pt = data_pt.cpu().numpy()
    array_pt = skimage.transform.resize(array_pt, (1, 240, 240), preserve_range=True, anti_aliasing=True)
    array_pt = (array_pt - array_pt.min()) / (array_pt.max() - array_pt.min())  # Normalize 0–1
    print("array_pt min and max:", array_pt.shape, array_pt.min(), array_pt.max())
else:
    array_pt = np.array(data_pt)
    array_pt = (array_pt - array_pt.min()) / (array_pt.max() - array_pt.min())

# For .nii.gz file
nii = nib.load('BraTS-GLI-00591-000-seg.nii.gz')
array_nii = nii.get_fdata()
array_nii = np.moveaxis(array_nii, -1, 0)  # Move the last axis to the front
#array_nii = (array_nii - array_nii.min()) / (array_nii.max() - array_nii.min())
print("array_nii:",array_nii.shape, array_nii.min(), array_nii.max())

nii2 = nib.load('BraTS-GLI-00474-001-t2w.nii.gz')
array_nii2 = nii2.get_fdata()
array_nii2 = np.moveaxis(array_nii2, -1, 0)  # Move the last axis to the front
#array_nii = (array_nii - array_nii.min()) / (array_nii.max() - array_nii.min())
print("array_nii2:",array_nii2.shape, array_nii2.min(), array_nii2.max())

nii3 = nib.load('BraTS-GLI-01771-000-t1c.nii.gz')
array_nii3 = nii3.get_fdata()
array_nii3 = np.moveaxis(array_nii3, -1, 0)  # Move the last axis to the front
#array_nii = (array_nii - array_nii.min()) / (array_nii.max() - array_nii.min())
print("array_nii3:",array_nii3.shape, array_nii3.min(), array_nii3.max())

nii4 = nib.load('BraTS-GLI-01705-000-t2w.nii.gz')
array_nii4 = nii4.get_fdata()
array_nii4 = np.moveaxis(array_nii4, -1, 0)  # Move the last axis to the front
#array_nii = (array_nii - array_nii.min()) / (array_nii.max() - array_nii.min())
print("array_nii3:",array_nii4.shape, array_nii4.min(), array_nii4.max())

# Launch napari viewer
viewer = napari.Viewer()
viewer.add_image(array_pt, name='PT Data')
#viewer.add_image(array_nii, name='00591-000-seg.nii.gz')
viewer.add_image(array_nii2[131], name='00474-001-t2w.nii.gz')
viewer.add_image(array_nii3, name='01771-000-t1c.nii.gz')
#viewer.add_image(array_nii4, name='01705-000-t2w.nii.gz')

napari.run()
