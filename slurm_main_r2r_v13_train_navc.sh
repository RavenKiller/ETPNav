#!/bin/bash
#SBATCH --job-name etpnav # 作业名
#SBATCH --partition L40 # amd 队列
#SBATCH -N 1 # amd 队列
#SBATCH -n 1 
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --output=r2r_v13_train_navc_eval_aug.out
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

flag2=" --exp_name release_r2r_v13_train_navc
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
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_v13_train_navc/ckpt.iter12000.pth
      IL.back_algo control
      "

# mode=eval
# port=2334
# case $mode in 
#       train)
#       echo "###### train mode ######"
#       torchrun --nproc_per_node=1 --master_port $port run.py $flag1
#       ;;
#       eval)
#       echo "###### eval mode ######"
#       torchrun --nproc_per_node=1 --master_port $port run.py $flag2
#       ;;
#       infer)
#       echo "###### infer mode ######"
#       torchrun --nproc_per_node=1 --master_port $port run.py $flag3
#       ;;
# esac


mode=eval
port=2334
CKPT_DIR="data/logs/checkpoints/release_r2r_v13_train_navc"

case $mode in 
    train)
        echo "###### train mode ######"
        torchrun --nproc_per_node=1 --master_port $port run.py $flag1
        ;;

    eval)
        echo "###### eval mode ######"
        for ckpt in "$CKPT_DIR"/ckpt.iter*.pth; do
            iter=$(basename "$ckpt" | grep -oP '\d+')
            if (( iter % 1000 == 0 )); then
                flag2=" --exp_name release_r2r_v13_train_navc
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
                    EVAL.CKPT_PATH_DIR $ckpt
                    EVAL.SPLIT val_unseen_navc(val_unseen)_1
                    IL.back_algo control
                "
                echo "Evaluating: $ckpt"
                torchrun --nproc_per_node=1 --master_port $port run.py $flag2
            fi
        done
        ;;

    infer)
        echo "###### infer mode ######"
        torchrun --nproc_per_node=1 --master_port $port run.py $flag3
        ;;
esac