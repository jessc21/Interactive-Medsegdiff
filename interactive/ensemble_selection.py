# phase 2 code
import napari
import torch as th
import numpy as np
import nibabel as nib
import os
import shutil
import sys
sys.path.append('/vol/bitbucket/yc3721/fyp/MedSegDiff')
from scripts.segmentation_sample import sample_once, create_argparser, create_model, load_data


def load_pt_as_np(path):
    return th.load(path).cpu().numpy()

def main(slice_idx):
    # ---- Load model and data
    args = create_argparser().parse_args()
    datal = load_data(args)
    model, diffusion = create_model(args)

    # ---- Load candidate masks
    masks = []
    num_candidates = 5  # or however many you generated
    for i in range(num_candidates):
        # generate masks by calling sample_once and then load them
        output_path = sample_once(int(slice_idx), i, args, datal, model, diffusion)
        # mask_path = f"./interaction_phase1_output/slice_{slice_idx}_mask_{i}.pt"
        mask = load_pt_as_np(output_path)
        masks.append(mask)

    # ---- Load original slice
    # todo: figure out what is original slice. t2w?
    folder = output_path.split('-')[0,1,2,3]
    filename = output_path.split('_')[0]
    nii = nib.load(f"./BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/{folder}/{filename}.nii.gz")
    original = nii.get_fdata()

    # ---- Start Napari viewer
    viewer = napari.Viewer()
    viewer.add_image(original, name="Original Slice")

    for idx, mask in enumerate(masks):
        viewer.add_labels(mask, name=f"Candidate Mask {idx}")

    print("Click on the mask name in the layer list to select your preferred one.")
    napari.run()

if __name__ == "__main__":
    slice_idx = input("Enter slice index: ")
    main(slice_idx)
