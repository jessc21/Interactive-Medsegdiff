#!/bin/bash
#SBATCH --job-name=medsegdiff
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --partition=gpgpu         # Target a big GPU like A40
#SBATCH --output=medseg_out_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yc3721@ic.ac.uk

which python
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# Load CUDA
source /vol/cuda/12.1.0/setup.sh

# Activate virtualenv
source /vol/bitbucket/yc3721/venv/medsegdiff-env/bin/activate

# Optional debug info
nvidia-smi
echo "Starting job on $(hostname)"
uptime

# Run training
python /vol/bitbucket/yc3721/fyp/MedSegDiff/scripts/segmentation_train.py \
    --data_name BRATS \
    --data_dir /vol/bitbucket/yc3721/fyp/MedSegDiff/BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
    --out_dir /vol/bitbucket/yc3721/fyp/MedSegDiff/output \
    --image_size 256 \
    --num_channels 128 \
    --class_cond False \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 50 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --lr 1e-4 \
    --batch_size 8 \
    --dpm_solver True \
    --resume_checkpoint /vol/bitbucket/yc3721/fyp/MedSegDiff/output/emasavedmodel_0.9999_010000.pt
