#!/bin/bash

# Define the dataset
DATASET='cifar10'
MODEL='vgg16'

# # Base Model
echo "Running Base Model..."
poetry run python train-vision.py --dataset $DATASET --model $MODEL --method 'base' --nepoch 7

# # Re-training Strategy (sampling)
echo "Running Re-training Strategy (sampling)..."
for epoch in 5 6 7 8 9
do
    poetry run python train-vision.py --dataset $DATASET --model $MODEL --method 'sampling' --sampling_nmodel 20 --nepoch $epoch
done

# # Dropout Strategy
echo "Running Dropout Strategy..."
poetry run python train-vision.py --dataset $DATASET --model $MODEL --nepoch 7 --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 50 --ndrp 5 --drp_max_ratio 0.008
poetry run python train-vision.py --dataset $DATASET --model $MODEL --nepoch 7 --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 50 --ndrp 5 --drp_max_ratio 0.1

# Compute Metrics
echo "Computing Metrics..."
poetry run python ../utils/compute_metrics.py --dataset $DATASET --model $MODEL --base_epoch 7 --method 'sampling' --sampling_nmodel 20 --epoch 5,6,7,8,9 --neps 6 --eps_max 0.05
poetry run python ../utils/compute_metrics.py --dataset $DATASET --model $MODEL --base_epoch 7 --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 50 --neps 6 --eps_max 0.05 --drp_max_ratio 0.008
poetry run python ../utils/compute_metrics.py --dataset $DATASET --model $MODEL --base_epoch 7 --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 50 --neps 6 --eps_max 0.05 --drp_max_ratio 0.1

# Ensemble Models
echo "Running Ensemble Models..."
poetry run python train-vision.py --dataset $DATASET --model $MODEL --nepoch 7 --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 10000 --ndrp 5 --drp_max_ratio 0.008
poetry run python train-vision.py --dataset $DATASET --model $MODEL --nepoch 7 --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 10000 --ndrp 5 --drp_max_ratio 0.1
poetry run python ../utils/ensemble.py --dataset $DATASET --model $MODEL --dropoutmethod 'bernoulli' --drp_nmodel 10000 --drp_max_ratio 0.008 --ensemble_size 1,2,5,10,20,50,100 --nensemble 100
poetry run python ../utils/ensemble.py --dataset $DATASET --model $MODEL --dropoutmethod 'gaussian' --drp_nmodel 10000 --drp_max_ratio 0.1 --ensemble_size 1,2,5,10,20,50,100 --nensemble 100

# Model Selection
echo "Running Model Selection..."
poetry run python train-model-selection.py --dataset $DATASET --model $MODEL --nretraining 10 --dropoutmethod 'bernoulli' --drp_nmodel 50 --drp_max_ratio 0.008
poetry run python train-model-selection.py --dataset $DATASET --model $MODEL --nretraining 10 --dropoutmethod 'gaussian' --drp_nmodel 50 --drp_max_ratio 0.1
poetry run python ../utils/compute_model_selection.py --dataset $DATASET --model $MODEL --dropoutmethod 'bernoulli' --nretraining 10 --nepoch 100
poetry run python ../utils/compute_model_selection.py --dataset $DATASET --model $MODEL --dropoutmethod 'gaussian' --nretraining 10 --nepoch 100

echo "All tasks completed."
