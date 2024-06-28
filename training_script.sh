#!/bin/bash
#SBATCH -J training
#SBATCH --gres=gpu:1
#SBATCH -w node5
#SBATCH -c 2
#SBATCH -N 1

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST 
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

#source /home/senmaoye/.bashrc

#cd jay/multitask

#conda activate torch

nvidia-smi

python train.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.06 --r 12 --alpha 16 --gamma 0.5 --beta 0.9999 --gpu 1 --data_path '/home/meiluzhu2/mm/6th_paper/data/total_data/myTensor_acc_combined_min_1.pt' --save_path 'setting1'
python train.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.06 --r 12 --alpha 16 --gamma 0.5 --beta 0.9999 --gpu 1 --data_path '/home/meiluzhu2/mm/6th_paper/data/total_data/myTensor_acc_combined_min_2.pt' --save_path 'setting2'
python train.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.06 --r 12 --alpha 16 --gamma 0.5 --beta 0.9999 --gpu 1 --data_path '/home/meiluzhu2/mm/6th_paper/data/total_data/myTensor_acc_combined_min_3.pt' --save_path 'setting3'
python train.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.06 --r 12 --alpha 16 --gamma 0.5 --beta 0.9999 --gpu 1 --data_path '/home/meiluzhu2/mm/6th_paper/data/total_data/myTensor_acc_combined_min_4.pt' --save_path 'setting4'
python train.py --epoch 100 --seed 10 --b 256 --lr 0.0001 --weight_d 0.06 --r 12 --alpha 16 --gamma 0.5 --beta 0.9999 --gpu 1 --data_path '/home/meiluzhu2/mm/6th_paper/data/total_data/myTensor_acc_combined_min_5.pt' --save_path 'setting5'
