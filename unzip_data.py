import os
import zipfile

# Paths to the downloaded zip files
train_zip = "/vol/bitbucket/yc3721/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
val_zip = "/vol/bitbucket/yc3721/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip"

# Target directory for extracted data
output_dir = "/vol/bitbucket/yc3721/fyp/MedSegDiff/BraTSdata"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to extract zip files
def unzip_file(zip_path, extract_to):
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction completed: {extract_to}")
    else:
        print(f"File not found: {zip_path}")

# Extract training and validation data
unzip_file(train_zip, output_dir)
unzip_file(val_zip, output_dir)

print("All files extracted successfully!")

