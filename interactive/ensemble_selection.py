# phase 2 code
import napari
import torch as th
import numpy as np
import nibabel as nib
import os
import shutil
import sys
import skimage.transform
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QThread, Signal, QObject
import csv
import datetime
sys.path.append('C:/Users/26364/Documents/IC/IC_Y4/FYP/MedSegDiff/')
from scripts.segmentation_sample import sample_once, create_argparser, create_model, load_data

class SamplingWorker(QObject):
    finished = Signal(list, list, int, np.ndarray)  # masks, output_paths, slice_id, original_slice

    def __init__(self, slice_idx, num_candidates, args, datal, model, diffusion):
        super().__init__()
        self.slice_idx = slice_idx
        self.num_candidates = num_candidates
        self.args = args
        self.datal = datal
        self.model = model
        self.diffusion = diffusion

    def run(self):
        masks = []
        output_paths = []

        for i in range(self.num_candidates):
            output_path = sample_once(self.slice_idx, i, self.args, self.datal, self.model, self.diffusion)
            mask = load_pt_as_np(output_path)
            masks.append(mask)
            output_paths.append(output_path)

        folder, filename, slice_id = parse_output_path(output_paths[0])
        original = nib.load(
            f"./BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/{folder}/{filename}.nii.gz"
        ).get_fdata()
        original = np.moveaxis(original, -1, 0)

        self.finished.emit(masks, output_paths, slice_id, original)

def load_pt_as_np(path):
    mask = th.load(os.path.join('interaction_phase2_output', path), map_location=th.device('cpu'))
    if isinstance(mask, th.Tensor):
        mask = mask.cpu().numpy()
    #mask = skimage.transform.resize(mask, (240, 240), preserve_range=True, anti_aliasing=True)
    #mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask

def parse_output_path(output_path):
    filename = os.path.basename(output_path)
    base = filename.split('_output')[0]
    filename_part, slice_part = base.split('_slice')
    slice_id = int(slice_part.split('.')[0])
    folder = '-'.join(filename_part.split('-')[:4])
    return folder, filename_part, slice_id



def main(slice_idx):
    worker = None
    thread = None

    # ---- Load model and data
    args = create_argparser().parse_args()
    datal = load_data(args)
    model, diffusion = create_model(args)
    slice_idx = int(slice_idx)


    # ---- Load candidate masks
    masks = []
    output_paths = []
    num_candidates = 1
    for i in range(num_candidates):
        output_path = sample_once(int(slice_idx), i, args, datal, model, diffusion)
        mask = load_pt_as_np(output_path)
        print(f"Loaded mask {i} from {output_path}, shape: {mask.shape}, min: {mask.min()}, max: {mask.max()}")
        masks.append(mask)
        output_paths.append(output_path)

    folder, filename, slice_id = parse_output_path(output_path)

    nii = nib.load(f"./BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/{folder}/{filename}.nii.gz")
    original = nii.get_fdata()
    original = np.moveaxis(original, -1, 0)
    print(f"Original shape: {original.shape}, slice_id: {slice_id}")

    viewer = napari.Viewer()
    viewer.add_image(original[slice_id], name="Original Slice")

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    layer_names = []

    for idx, mask in enumerate(masks):
        layer = viewer.add_image(
            mask,
            name=f"Candidate Mask {idx}",
            opacity=0.15,
            colormap=colors[idx % len(colors)],
            blending="additive"
        )
        layer_names.append(layer.name)

    # ------ BUTTON LOGIC ---------
    def save_selected_mask():
        selected_layer = viewer.layers.selection.active
        if selected_layer is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please select a mask layer before clicking Save.")
            msg.setWindowTitle("No Layer Selected")
            msg.exec_()
            return

        selected_name = selected_layer.name
        if not selected_name.startswith("Candidate Mask"):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please select a mask layer before clicking Save.")
            msg.setWindowTitle("No Layer Selected")
            msg.exec_()
            return

        chosen_idx = int(selected_name.split()[-1])
        chosen_mask_path = output_paths[chosen_idx]
        chosen_output_dir = os.path.join('interaction_phase2_output', 'user_selected')
        os.makedirs(chosen_output_dir, exist_ok=True)

        shutil.copy(
            os.path.join('interaction_phase2_output', os.path.basename(chosen_mask_path)),
            os.path.join(chosen_output_dir, os.path.basename(chosen_mask_path))
        )
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Chosen mask saved to {chosen_output_dir}")
        msg.setWindowTitle("Mask Saved")
        msg.exec_()

        # ----- CSV LOGGING -----
        log_path = os.path.join(chosen_output_dir, "user_choices.csv")
        fieldnames = ["slice_idx", "chosen_mask_idx", "mask_filename", "time"]

        row = {
            "slice_idx": slice_idx,
            "chosen_mask_idx": chosen_idx,
            "mask_filename": os.path.basename(chosen_mask_path),
            "time": datetime.datetime.now().isoformat()
        }

        file_exists = os.path.isfile(log_path)
        with open(log_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"Logged choice to {log_path}")
    
    def run_next_slice():
        nonlocal slice_idx, worker, thread
        slice_idx += 155

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Sampling for slice {slice_idx}...")
        msg.setWindowTitle("Sampling")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()  # Non-blocking
        #viewer.status = f"Sampling for slice {slice_idx}..."

        # Clear previous mask layers
        for name in layer_names:
            if name in viewer.layers:
                viewer.layers.remove(name)

        # Start background thread
        worker = SamplingWorker(slice_idx, num_candidates, args, datal, model, diffusion)
        thread = QThread()
        worker.moveToThread(thread)

        # When thread starts, call worker.run
        thread.started.connect(worker.run)

        # When worker finishes, call update_viewer and clean up
        def update_viewer(masks_new, output_paths_new, slice_id_new, original_new):
            nonlocal masks, output_paths

            masks = masks_new
            output_paths = output_paths_new

            viewer.layers["Original Slice"].data = original_new[slice_id_new]

            for idx, mask in enumerate(masks):
                layer = viewer.add_image(
                    mask,
                    name=f"Candidate Mask {idx}",
                    opacity=0.15,
                    colormap=colors[idx % len(colors)],
                    blending="additive"
                )
                layer_names.append(layer.name)

            #viewer.status = "Sampling completed."
            thread.quit()
            thread.wait()

        worker.finished.connect(update_viewer)

        thread.start()

    instruction = QLabel("Select a mask layer and click 'Save Selected Mask' to confirm your choice.")
    instruction.setWordWrap(True)
    viewer.window.add_dock_widget(instruction, area="right")

    # Add button to Napari
    button = QPushButton("Save Selected Mask")
    button.clicked.connect(save_selected_mask)
    viewer.window.add_dock_widget(button, area="right")

    # Add Next Slice button
    next_button = QPushButton("Next Slice")
    next_button.clicked.connect(run_next_slice)
    viewer.window.add_dock_widget(next_button, area="right")


    napari.run()


if __name__ == "__main__":
    slice_idx = input("Enter slice index: ")
    main(slice_idx)
