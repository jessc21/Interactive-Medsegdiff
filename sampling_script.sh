# Load CUDA
source /vol/cuda/12.1.0/setup.sh

# Activate virtualenv
source /vol/bitbucket/yc3721/venv/medsegdiff-env/bin/activate

# Optional debug info
nvidia-smi
echo "Starting job on $(hostname)"
uptime

# Run sampling
python scripts/segmentation_sample.py \
    --data_name BRATS \
    --data_dir /vol/bitbucket/yc3721/fyp/MedSegDiff/BraTSdata/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData \
    --out_dir /vol/bitbucket/yc3721/fyp/MedSegDiff/samples_interaction_test_shuffle \
    --model_path /vol/bitbucket/yc3721/fyp/MedSegDiff/output/emasavedmodel_0.9999_065000.pt \
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
    --dpm_solver True \
    --num_ensemble 5
