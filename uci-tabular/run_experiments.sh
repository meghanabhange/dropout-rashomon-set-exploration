#!/bin/bash

# Define the dataset
DATASET='credit-approval'

# # Base Model
echo "Running Base Model..."
poetry run python train-tabular.py --dataset $DATASET --method 'base'

# # Re-training Strategy (sampling)
echo "Running Re-training Strategy (sampling)..."
for epoch in 20 25 30 35 40 45 50 55
do
    poetry run python train-tabular.py --dataset $DATASET --method 'sampling' --sampling_nmodel 100 --nepoch $epoch
done

# # Dropout Strategy
echo "Running Dropout Strategy..."
poetry run python train-tabular.py --dataset $DATASET --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 100 --drp_max_ratio 0.2
poetry run python train-tabular.py --dataset $DATASET --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 100 --drp_max_ratio 0.6

# # AWP Strategy
echo "Running AWP Strategy..."
poetry run python train-tabular.py --dataset $DATASET --method 'awp' --awp_eps 0.000,0.004,0.008,0.012,0.016,0.020,0.024,0.028,0.032,0.036,0.040

# Compute Metrics
echo "Computing Metrics..."
poetry run python ../utils/compute_metrics.py --dataset $DATASET --method 'sampling' --sampling_nmodel 100 --epoch 20,25,30,35,40,45,50,55
poetry run python ../utils/compute_metrics.py --dataset $DATASET --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 100 --drp_max_ratio 0.2
poetry run python ../utils/compute_metrics.py --dataset $DATASET --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 100 --drp_max_ratio 0.6
poetry run python ../utils/compute_metrics.py --dataset $DATASET --method 'awp' --awp_eps 0.000,0.004,0.008,0.012,0.016,0.020,0.024,0.028,0.032,0.036,0.040

# Ensemble Models
echo "Running Ensemble Models..."
poetry run python train-tabular.py --dataset $DATASET --method 'dropout' --dropoutmethod 'bernoulli' --drp_nmodel 10000 --drp_max_ratio 0.2
poetry run python train-tabular.py --dataset $DATASET --method 'dropout' --dropoutmethod 'gaussian' --drp_nmodel 10000 --drp_max_ratio 0.6
poetry run python ../utils/ensemble.py --dataset $DATASET --dropoutmethod 'bernoulli' --drp_nmodel 10000 --drp_max_ratio 0.2 --ensemble_size 1,2,5,10,20,50,100 --nensemble 100
poetry run python ../utils/ensemble.py --dataset $DATASET --dropoutmethod 'gaussian' --drp_nmodel 10000 --drp_max_ratio 0.6 --ensemble_size 1,2,5,10,20,50,100 --nensemble 100

# Model Selection
echo "Running Model Selection..."
poetry run python train-model-selection.py --dataset $DATASET --nretraining 10 --dropoutmethod 'bernoulli' --drp_nmodel 100 --drp_max_ratio 0.2
poetry run python train-model-selection.py --dataset $DATASET --nretraining 10 --dropoutmethod 'gaussian' --drp_nmodel 100 --drp_max_ratio 0.6
poetry run python ../utils/compute_model_selection.py --dataset $DATASET --dropoutmethod 'bernoulli' --nretraining 10 --nepoch 100
poetry run python ../utils/compute_model_selection.py --dataset $DATASET --dropoutmethod 'gaussian' --nretraining 10 --nepoch 100

echo "All tasks completed."
