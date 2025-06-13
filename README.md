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

## Installation

**Dependencies:**

- Python ≥ 3.8
- PyTorch ≥ 1.12
- Napari
- NumPy, SciPy, scikit-image
- [maxflow](https://github.com/pmneila/PyMaxflow)
- OpenCV

**Clone and set up:**

```bash
git clone https://github.com/yourusername/interactive-medseg-diffusion.git
cd interactive-medseg-diffusion
pip install -r requirements.txt
