##### GIN fine-tuning
#split=scaffold
#dataset=$1
#
#CUDA_VISIBLE_DEVICES=0
#for runseed in 0 1 2 3 4 5 6 7 8 9
#do
#model_file=${unsup}
##CUDA_VISIBLE_DEVICES=5 python finetune.py --input_model_file models_pgcl_sr/all_2.pth --split scaffold --runseed 0 --gnn_type gin --dataset tox21 --lr 1e-3 --epochs 100
#CUDA_VISIBLE_DEVICES=5 python finetune_virtual.py --input_model_file models_pgcl_sr/all_2.pth --split scaffold --runseed 0 --gnn_type gin --dataset tox21 --lr 1e-3 --epochs 80
#
#done
#
##CUDA_VISIBLE_DEVICES=5 python finetune.py --input_model_file models_graphcl/masking_virtual80.pth --split scaffold --runseed 0 --gnn_type gin --dataset tox21 --lr 1e-3 --epochs 100

split=scaffold
dataset=$1

CUDA_VISIBLE_DEVICES=0
for dataset in clintox hiv
do
for runseed in 0 1 2 3 4 5 6 7 8 9
do
model_file=${unsup}
#CUDA_VISIBLE_DEVICES=5 python finetune.py --input_model_file models_pgcl_sr/all_2.pth --split scaffold --runseed 0 --gnn_type gin --dataset tox21 --lr 1e-3 --epochs 100
CUDA_VISIBLE_DEVICES=4 python finetune_virtual.py --input_model_file models_pgcl_sr/new_20.pth --split scaffold --runseed 0 --gnn_type gin --dataset $dataset --lr 5e-4 --epochs 50
done
done