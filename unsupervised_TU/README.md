# Prototypical Graph Contrastive Learning

The code of PGCL in the unsupervised setting.

## Dependencies

We develop this project with `Python3.6, cuda10.1` and  following Python packages:

```
Pytorch                   1.4.0
torch-cluster             1.5.4                    
torch-geometric           1.6.1                    
torch-scatter             2.0.4                    
torch-sparse              0.6.1                    
torch-spline-conv         1.2.0 
```
Then, you need to create a directory for recoreding finetuned results to avoid errors:

```
mkdir logs
```

## Training & Evaluation

### For PGCL
```
./go_PGCL.sh $GPU_ID $DATASET_NAME $AUGMENTATION1  $AUGMENTATION2 $GCN_LAYER $LEARNING_RATE $EPOCHS $NMB_ROTOTYPES
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```dnodes or pedges```. `$GCN_LAYER` is the gcn layer (default: 3). `$LEARNING_RATE` is the learning rate (default: 0.01). `$EPOCHS` is the max_epochs (default: 20) and `$NMB_ROTOTYPES` is the number of prototypes (default: 10).


## Acknowledgements

1. The backbone implementation is reference to: https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised.
2. The augmentation implementation is reference to: https://github.com/Shen-Lab/GraphCL.
