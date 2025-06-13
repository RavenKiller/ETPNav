#!/bin/bash
#SBATCH --job-name etpnav # 作业名
#SBATCH --partition L40 # amd 队列
#SBATCH -N 1 # amd 队列
#SBATCH -n 1 
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --output=z_log.out
#SBATCH --mail-type=end
#SBATCH --mail-user=1161188300@qq.com

source ~/.bashrc
conda activate habitat21
cd ~/hzt/clash
nohup ./clash-linux-amd64-v1.2.0 -f 1725849302984.yml &
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
cd ~/hzt/ETPNav
CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash train 2333