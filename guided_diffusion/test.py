from torch.utils.data import DataLoader
from bratsloader import BRATSDataset3D

ds = BRATSDataset3D("/vol/bitbucket/yc3721/fyp/MedSegDiff/BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/", test_flag=False)
# loader = DataLoader(ds, batch_size=1, shuffle=False)
# next(iter(loader))  # This triggers __getitem__() and prints stats

image, label, path = ds[999]

# for i, name in enumerate(['t1c', 't1n', 't2f', 't2w']):
#     print(f"{name} min={image[i].min().item():.2f}, max={image[i].max().item():.2f}")

# print("Label unique values:", label.unique())

print(f"Total dataset samples: {len(ds)}")
first_sample = ds[0]
print(f"First sample shape: {first_sample[0].shape}, Label shape: {first_sample[1].shape}")
