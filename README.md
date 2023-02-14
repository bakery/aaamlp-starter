# AAAMLP starter

Starter kit for projects from [Approaching Almost Any Machine Learning problem](https://github.com/abhishekkrthakur/approachingalmost) book by [Abhishek Thakur](https://github.com/abhishekkrthakur)

---

## Note 👷

This starter is very much work in progress. I am updating key modules as I am making my way through the book.

## Prereqs

- install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- `conda create -n NAME_FOR_YOUR_CONDA_ENV python=3.7.6` (if you are on M1/M2, you might need to run this before `conda config --env --set subdir osx-64`)
- `conda activate NAME_FOR_YOUR_CONDA_ENV`
- `conda env create -f environment.yml`
- OR `conda env create -f environment_osx.yml` on Mac
- `conda activate ml`
- download `mnist_train.csv` from [Abhishek's Kaggle](https://www.kaggle.com/datasets/abhishek/aaamlp), and save it to `input` directory

## Getting started

This template gets you started with a basic skeleton for your ML app.

### Creating folds

- `cd src && python create_folds.py --folds 10` (default is 5 folds)

### Adding/editing available models

`model_dispatcher.py` contains a dictionary with models available in your application

### Training your model

To train your model, use 

`python train.py --fold 0 --model desision_tree_entropy`
