# phase 1 code
import os
import shutil
import sys
sys.path.append(".")
from scripts.segmentation_sample import sample_once, create_argparser, create_model, load_data

def user_feedback_loop(slice_idx, max_attempts=5):
    args = create_argparser().parse_args()
    datal = load_data(args)
    model, diffusion = create_model(args)

    for attempt in range(max_attempts):
        print(f"\nSampling attempt {attempt + 1} for slice {slice_idx}")
        output_path = sample_once(int(slice_idx), attempt, args, datal, model, diffusion)
        user_input = input("Is the segmentation acceptable? (yes/no): ").strip().lower()
        if user_input == "yes":
            print("Accepted by user.")
            break
        else:
            print("Resampling...")

    else:
        print("Max attempts reached. Please consider manual editing.")

if __name__ == "__main__":
    slice_idx = input("Enter slice Idx to segment: ")
    user_feedback_loop(slice_idx)
