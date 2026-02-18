#!/bin/bash
#SBATCH --job-name=af_pipeline
#SBATCH --partition=dhvi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=covid_%j.out
#SBATCH --error=covid_%j.err

source activate gromacs_env

RF_DIR=/cwork/hsb26/gefh1_designs/20250924_140319/RFD/outputs/
AF_DIR=/cwork/hsb26/gefh1_designs/20250924_140319/AF/outputs/ 
OUT_DIR=/cwork/pkk13/pipeline/

python3.9 -u lyze.py \
  --rf_base_dir  "${RF_DIR}" \
  --af_base_dir  "${AF_DIR}" \
  --output_dir   "${OUT_DIR}" \
  --generate_pdf \
  --pdf_name GEFDEC8.pdf
