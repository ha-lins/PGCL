*Code to be further cleaned...*

## Dependencies & Dataset

Please refer to https://github.com/snap-stanford/pretrain-gnns#installation for environment setup and https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset.

```
pip install torch-scatter==1.1.2 -f https://pytorch-geometric.com/whl/torch-1.0.1+cu90.html
pip install torch-sparse==0.2.4 -f https://pytorch-geometric.com/whl/torch-1.0.1+cu90.html
pip install torch-cluster==1.2.4  -f https://pytorch-geometric.com/whl/torch-1.0.1+cu90.html --no-cache-dir 
pip install torch-spline-conv==1.0.6 -f https://pytorch-geometric.com/whl/torch-1.0.1+cu90.html
pip install torch-geometric==1.0.3
```
If you cannot manage to install the old torch-geometric version, one alternative way is to use the new one (maybe ==1.6.0) and make some modifications based on this issue https://github.com/snap-stanford/pretrain-gnns/issues/14.
This might leads to some inconsistent results with those in the paper.

## Training & Evaluation
### Pre-training: ###
```
cd ./chem
python pgcl_sr.py --aug1 dropN --aug2 permE
```

### Finetuning: ###
```
cd ./chem
./run.sh
```
Results will be recorded in ```result.log```.


## Acknowledgements

The training and evaluation pipeline is referred from GraphCL [Neurips 2020]. The backbone implementation is reference to https://github.com/snap-stanford/pretrain-gnns.
