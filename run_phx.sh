#!/bin/bash

# Configure the resources required
#SBATCH -M volta
#SBATCH -N 1                                                   # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 4                                                   # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=24:00:00                                        # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1                                           # generic resource required (here requires 4 GPUs)
#SBATCH --mem=32GB                                             # specify memory required per node (here set to 16 GB)

# Configure notifications
#SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=qi.chen04@adelaide.edu.au                # Email to which notification will be sent

nvidia-smi -l > nv-smi.log 2>&1 &

export MASTER_ADDR=localhost
export MASTER_PORT=12345

# ====================================================
# input parameters
# ====================================================
# (Step 1)
data_path=/hpcfs/users/a1796450/Evaluation/GenerativeResults5k/glide
custom_data_filename=custom_data.txt
custom_data_code_filename=custom_data_code.txt

# (Step 2)
custom_data=${data_path}/${custom_data_filename}
custom_data_code=${data_path}/${custom_data_code_filename}

# (Step 3)
user_dir=ofa_module
bpe_dir=utils/BPE

# image_gen_large_best.pt and image_gen_base_best.pt
path=./checkpoints/image_gen_large_best.pt
selected_cols=0,1,2,3
split='test'
VQGAN_MODEL_PATH=./checkpoints/vqgan/last.ckpt
VQGAN_CONFIG_PATH=./checkpoints/vqgan/model.yaml
CLIP_MODEL_PATH=./checkpoints/clip/ViT-B-16.pt
GEN_IMAGE_PATH=./results/image_gen

image_path=${data_path}/image
output_path=${data_path}/output
score_filepath=${data_path}/scores_fin.txt

# mode for clip score {base-p16-224, base-p32-224, large-p14-224, large-p14-336}
clip_mode=base-p16-224


/hpcfs/users/a1796450/anaconda3/envs/evaluation/bin/python myevaluation.py \
  --data_path=${data_path} \
  --custom_data_filename=${custom_data_filename} \
  --file=${custom_data} \
  --outputs=${custom_data_code} \
  --selected_cols 0,1,2,3 \
  --code_image_size 256 \
  --vq_model vqgan \
  --vqgan_model_path ./checkpoints/vqgan/last.ckpt \
  --vqgan_config_path ./checkpoints/vqgan/model.yaml \
  data=${custom_data_code} \
  --path=${path} \
  --user-dir=${user_dir} \
  --task=image_gen \
  --batch-size=1 \
  --log-format=simple --log-interval=1 \
  --seed=42 \
  --gen-subset=${split} \
  --beam=1 \
  --min-len=1024 \
  --max-len-a=0 \
  --max-len-b=1024 \
  --sampling-topk=1 \
  --temperature=1.0 \
  --code-image-size=256 \
  --constraint-range=50265,58457 \
  --fp16 \
  --num-workers=0 \
  --model-overrides="{\"data\":\"${custom_data_code}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"clip_model_path\":\"${CLIP_MODEL_PATH}\",\"vqgan_model_path\":\"${VQGAN_MODEL_PATH}\",\"vqgan_config_path\":\"${VQGAN_CONFIG_PATH}\",\"gen_images_path\":\"${GEN_IMAGE_PATH}\", \"image_path\":\"${image_path}\", \"output_path\":\"${output_path}\", \"score_filepath\":\"${score_filepath}\", \"clip_mode\":\"${clip_mode}\"}"

