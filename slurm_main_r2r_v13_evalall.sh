#!/bin/bash
#SBATCH --job-name eval # 作业名
#SBATCH --partition L40 # amd 队列
#SBATCH -N 1 # amd 队列
#SBATCH -n 1 
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --output=evalall_r2r_v13_train_navc(train)_1_finetune2.out
#SBATCH --mail-type=end
#SBATCH --mail-user=1161188300@qq.com

source ~/.bashrc
conda activate habitat21
cd ~/hzt/clash
nohup ./clash-linux-amd64-v1.2.0 -f 1725849302984.yml &
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
cd ~/hzt/ETPNav


export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()');
exp_name="r2r_v13_train_navc(train)_1_finetune"
for eval_split in "val_unseen" "navc(val_unseen)_1" "navc(val_unseen)_2" "navc(val_unseen)_3" "val_unseen_navc(val_unseen)_1" "val_unseen_navc(val_unseen)_2" "val_unseen_navc(val_unseen)_3"
do
      flag2=" --exp_name ${exp_name}
            --run-type eval
            --exp-config run_r2r/iter_train.yaml
            SIMULATOR_GPU_IDS [0]
            TORCH_GPU_IDS [0]
            GPU_NUMBERS 1
            NUM_ENVIRONMENTS 6
            TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
            TASK_CONFIG.TASK.NDTW.GT_PATH data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
            TASK_CONFIG.TASK.SDTW.GT_PATH data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
            TASK_CONFIG.DATASET.DATA_PATH data/datasets/R2R_VLNCE_v1-3_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz
            EVAL.CKPT_PATH_DIR data/logs/checkpoints/${exp_name}/ckpt.iter12000.pth
            EVAL.SPLIT ${eval_split}
            IL.back_algo control
            "
      torchrun --nproc_per_node=1 --master_port $port run.py $flag2
      flag2=" --exp_name ${exp_name}
            --run-type eval
            --exp-config run_r2r/iter_train.yaml
            SIMULATOR_GPU_IDS [0]
            TORCH_GPU_IDS [0]
            GPU_NUMBERS 1
            NUM_ENVIRONMENTS 6
            TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
            TASK_CONFIG.TASK.NDTW.GT_PATH data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
            TASK_CONFIG.TASK.SDTW.GT_PATH data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz
            TASK_CONFIG.DATASET.DATA_PATH data/datasets/R2R_VLNCE_v1-3_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz
            EVAL.CKPT_PATH_DIR data/logs/checkpoints/${exp_name}/ckpt.iter15000.pth
            EVAL.SPLIT ${eval_split}
            IL.back_algo control
            "
      torchrun --nproc_per_node=1 --master_port $port run.py $flag2
      echo "Finish ${exp_name} on ${eval_split}"
      echo "\n"
      echo "\n"
done
