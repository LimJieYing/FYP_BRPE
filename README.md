# Blind room parameter estimation from reverberant speech using machine learning
### By Lim Jie Ying

This project generates synthetic reverberant speech data, extracts features, and trains an LSTM model to estimate RT60 

## dataset download

Link to download the dataset: https://www.openslr.org/12 

for current setup and path:


train-clean-100.tar.gz dataset is used for training

test-clean.tar.gz dataset is used for evaluation

## Data generation pipeline 
To run the pipeline, run 'run_pipeline.py' script to automatically run the stages

## Training
To train the LSTM model, run train_lstm.py

Note that GPU acceleration should be used while training.


ensure your computer  has a CUDA-compatible GPU and PyTorch is installed with CUDA support. 
The code will automatically use the GPU if available.


## create evaluation set
To create the evaluation data, run 'run_eval_pipeline.py'
note: this pipeline reuses convolusion and feature extraction from data generation pipeline

## evaluation
To evaluate, run 'evaluate_test.py'

