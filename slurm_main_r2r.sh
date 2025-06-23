#!/bin/bash
#SBATCH --job-name etpnav # 作业名
#SBATCH --partition L40 # amd 队列
#SBATCH -N 1 # amd 队列
#SBATCH -n 1 
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --output=r2r_iter12000.out
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

flag1="--exp_name release_r2r
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 8
      IL.iters 15000
      IL.lr 1e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 3000
      IL.load_from_ckpt False
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt
      "

flag2=" --exp_name release_r2r
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 6
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      IL.back_algo control
      "

flag3="--exp_name release_r2r
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 6
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "

mode=eval
port=2333
case $mode in 
      train)
      echo "###### train mode ######"
      torchrun --nproc_per_node=1 --master_port $port run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      torchrun --nproc_per_node=1 --master_port $port run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      torchrun --nproc_per_node=1 --master_port $port run.py $flag3
      ;;
esac