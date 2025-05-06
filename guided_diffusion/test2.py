import nibabel as nib
import numpy as np

f = nib.load("/vol/bitbucket/yc3721/fyp/MedSegDiff/BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz")
data = f.get_fdata()
print("Shape:", data.shape)
print("Min:", np.min(data), "Max:", np.max(data))
