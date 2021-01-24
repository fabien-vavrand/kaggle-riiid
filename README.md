# Kaggle "Riiid Answer Correctness Prediction" competition
Competition overview: https://www.kaggle.com/c/riiid-test-answer-prediction/overview/description

This repository contains the code used to generate and submit the solution, which ranked **56th** with a **80.2%** ROC AUC score.
The best performing model is based on a blending of a Lightgbm, a Catboost, and a Keras MLP model.

The repository also contains an encoder-decoder transformer based model, inspired by the [Saint+ paper](https://arxiv.org/abs/2010.12042) which scored **79.6%**, but was not integrated in the final solution due to the submission notebook running time constraint of 9 hours.

## Running the code
- Create a new `Riiid` folder, and a sub folder `data`
- Download and save the competition data in the newly created `data` folder
- Set the `RIIID_PATH` environment variable to the `Riiid` folder path
- Run `scripts/build_validation.py`
- Run `scripts/train.py`

The training is configured to be performed on a small subset of the data (30k users), on a 16GB machine. Training the model on the full dataset requires 256GB of RAM, and was performed on an AWS EC2 instance by running `aws/train.py` (running this script requires an AWS account, credentials, and the [AWS Doppel package](https://github.com/fabien-vavrand/aws-doppel)).

The Saint+ like Transformer model can be trained by running `scripts/train_saint.py`. On the full dataset, features where generated using an AWS 128GB EC2 instance and the model was trained on a Kaggle TPUv3 notebook for 2 hours.