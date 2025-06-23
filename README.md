# Interactive Medical Image Segmentation using Diffusion Models

This project presents an interactive segmentation pipeline for medical images, combining a latent diffusion-based segmentation backbone with user-driven refinement modules. The system supports human-in-the-loop decision-making via multi-sample mask selection, manual contour editing (BEAS), and brush-based corrections (Graph Cut), all through an intuitive Napari interface.

## Features

- Generative segmentation using latent diffusion models (MedSegDiff)
- Stochastic sampling with user selection of best mask
- BEAS-based contour editing for fine-grained boundary correction
- Graph Cut brush refinement for flexible soft corrections
- Modular architecture designed for clinical usability
- Lightweight CPU-based inference and interaction
- Built-in Napari GUI for visualization and interaction

---
## Demo

A short demo of the interactive segmentation pipeline:

▶️ [Click here to watch the demo (MP4)](./demo/your_demo_video.mp4)

---
## How to Run

Due to GitHub's file size limits, the pretrained diffusion model used for mask generation is not included in this repository. As such, running the sampling pipeline end-to-end is not directly supported unless you retrain the model yourself.

However, we provide a sample segmentation result for testing purposes. You can still test and interact with the BEAS and Graph Cut refinement modules independently:

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/interactive-medseg-diffusion.git
cd interactive-medseg-diffusion
```

### 2. Install the required dependencies
```bash 
pip install -r requirements.txt
```

### 3. Navigate to the interactive folder
```bash 
cd interactive
```

### 4. Run BEAS refinement demo
```bash 
python beas_test.py
```

### 5. Run Graph Cut refinement demo
```bash 
python graphcut_test.py
```

These scripts will load a test segmentation mask and launch the Napari GUI for interactive refinement.

## Installation

**Dependencies:**

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- Napari  
- NumPy  
- SciPy  
- scikit-image  
- [maxflow](https://github.com/pmneila/PyMaxflow)  
- OpenCV  
