from guided_diffusion.bratsloader import BRATSDataset3D

ds = BRATSDataset3D("/vol/bitbucket/yc3721/fyp/MedSegDiff/BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", None, test_flag=False)
print(f"Total dataset samples: {len(ds)}")
first_sample = ds[0]
print(f"First sample shape: {first_sample[0].shape}, Label shape: {first_sample[1].shape}")

