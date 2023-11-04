# cgiar-eyes-on-the-ground-challenge
https://zindi.africa/competitions/cgiar-eyes-on-the-ground-challenge
# Split kfold
```
cd process_data
python check_group.py
```
# Training
Train regession model using BCE loss
```
python code/train.py --exp exp_16
```
Generate softlabel
```
python code/generate_softlabel.py
```
Train model with softlabel
```
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 code/finetune_softlabel.py --exp exp_17
```
# Predict
Infer single model (modify weight path and image size in code)
```
python code/predict.py
```
Ensemble 3 models
```
python code/ensemble.py
```
