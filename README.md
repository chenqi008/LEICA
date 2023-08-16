# LEICA
The official code for "LEICA: Likelihood-Based Text-to-Image Evaluation with Credit Assignment"

## Download Checkpoints
Download checkpoints.zip ([Baidu Netdisk](https://pan.baidu.com/s/1U-YASHgWu4LoMw8wufmKlQ?pwd=leic)[提取码:leic], [google drive]) and unzip it in the root directory of this project.
```
unzip checkpoints.zip
```
Download OFA pre-trained checkpoints from [Pre-trained checkpoint (OFA-large)(5.9GB)](https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_large.pt) and [Pre-trained checkpoint (OFA-Base)(5.9GB))](https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_base.pt).
```
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_large.pt
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_base.pt
```
Then put them into checkpoints dir.
```
mv image_gen_* ./checkpoints
```

## Data Preparation

## Run Evaluation
Run the following command to evaluate with OFA.
```
./run.sh ofa
```