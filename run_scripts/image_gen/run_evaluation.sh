#!/usr/bin/env bash

# ====================================================
# input parameters
# ====================================================
# (Step 1)
data_path=./toy_data/coco_karpathy_split/gaussian
custom_data_filename=custom_data.txt
custom_data_code_filename=custom_data_code.txt

# (Step 2)
custom_data=${data_path}/${custom_data_filename}
custom_data_code=${data_path}/${custom_data_code_filename}

# (Step 3)
user_dir=../../ofa_module
bpe_dir=../../utils/BPE

path=../../checkpoints/image_gen_large_best.pt
selected_cols=0,2,3
split='test'
VQGAN_MODEL_PATH=../../checkpoints/vqgan/last.ckpt
VQGAN_CONFIG_PATH=../../checkpoints/vqgan/model.yaml
CLIP_MODEL_PATH=../../checkpoints/clip/ViT-B-16.pt
GEN_IMAGE_PATH=../../results/image_gen

# # ====================================================
# # (Step 1)
# # process the original image to base64 format
# # and yield a custom_data.txt file
# # ====================================================

# CUDA_VISIBLE_DEVICES=0 python data_processing.py --data_path=${data_path} --custom_data_filename=${custom_data_filename}


# # ====================================================
# # (Step 2)
# # generate image tokens from custom_data.txt
# # and yield a custom_data_code.txt file
# # ====================================================
# # for text-image paired data, each line of the given input file should contain these information (separated by tabs):
# # input format
# #   uniq-id, image-id, image base64 string and text
# # input example
# #   162365  12455 /9j/4AAQSkZJ....UCP/2Q==  two people in an ocean playing with a yellow frisbee.
# #
# # output format
# #   uniq-id, image-id, text and code
# # output example
# #   162364 12455 two people in an ocean playing with a yellow frisbee.  6288 4495 4139...4691 4844 6464

# CUDA_VISIBLE_DEVICES=0 python generate_code.py \
#   --file=${custom_data} \
#   --outputs=${custom_data_code} \
#   --selected_cols 0,1,2,3 \
#   --code_image_size 256 \
#   --vq_model vqgan \
#   --vqgan_model_path ../../checkpoints/vqgan/last.ckpt \
#   --vqgan_config_path ../../checkpoints/vqgan/model.yaml


# ====================================================
# (Step 3)
# calculate score based on text and image tokens
# ====================================================
# It may take a long time for the full evaluation. You can sample a small split from the full_test split.
# But please remember that you need at least thousands of images to compute FID and IS, otherwise the resulting scores
# might also no longer correlate with visual quality.

CUDA_VISIBLE_DEVICES=0 python ../../evaluate.py \
  ${custom_data_code} \
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
  --model-overrides="{\"data\":\"${custom_data_code}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"clip_model_path\":\"${CLIP_MODEL_PATH}\",\"vqgan_model_path\":\"${VQGAN_MODEL_PATH}\",\"vqgan_config_path\":\"${VQGAN_CONFIG_PATH}\",\"gen_images_path\":\"${GEN_IMAGE_PATH}\"}"
