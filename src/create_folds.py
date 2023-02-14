import argparse
import pandas as pd
import numpy as np
from sklearn import model_selection

import config

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--folds", type=int, default=config.DEFAULT_NUMBER_OF_FOLDS)

  args = parser.parse_args()

  print(f"Creating {args.folds} folds for {config.TRAINING_FILE} ...")

  df = pd.read_csv(config.TRAINING_FILE)

  df["kfold"] = -1
  
  # randomize
  df = df.sample(frac=1).reset_index(drop=True)
  

  # ⚠️ Here is where we choose how to fold
  
  # ** Stratified K-Fold
  #
  #    👍 Rule of thumb: If it's a standard classification problem, choose stratified k-fold
  #
  #    Similar as K-Fold but it respects proportions between different categories: e.g in a set of
  #    90% positive and 10% negative samples, we want to preserve this very skewed ratio when dividing
  #    our set into the folds. Usage:
  #   
  #    kf = model_selection.StratifiedKFold(n_splits=args.folds)
  #    for fold, (t_, v_) in enumerate(kf.split(X=df, y=df[config.TARGET_COL_NAME].values)):
  #        df.loc[v_, 'kfold'] = fold


  # ** Hold out based valiation
  #       
  #    👍 Rule of thumb: If you have a large amount of data (e.g. 1 million samples) or you are dealing with time-series,
  #                      use hold out based validation
  #
  #    -> use stratified k-fold to produce 10 folds 
  #    -> keep one of those as hold-out 
  #    -> train on 9 remaining folds
  #    -> calculate loss, accuracy and other metricsthe hold-out fold 

  
  # ** Stratified K-fold for regression
  #
  #    -> divide target into bins (see how bins work here => https://pandas.pydata.org/docs/reference/api/pandas.cut.html)
  #    -> if you have a lot of samples (>10k), just use 10-20 bins
  #    -> if not, use Sturge's Rule to calc how many bins you want
  #           Number_Of_Bins = 1 + log2(Number_Of_Samples)
  #     
  #    Usage:
  #
  #    num_bins = int(np.floor(1 + np.log2(len(df))))
  #    df.loc[:, "bins"] = pd.cut(df[config.TARGET_COL_NAME], bins=num_bins, labels=False) 
  #    kf = model_selection.StratifiedKFold(n_splits=args.folds)
  #    for fold, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
  #        df.loc[v_, 'kfold'] = fold

  kf = model_selection.StratifiedKFold(n_splits=args.folds)
  for fold, (t_, v_) in enumerate(kf.split(X=df, y=df[config.TARGET_COL_NAME].values )):
      df.loc[v_, 'kfold'] = fold

      
  df.to_csv(config.TRAINING_FILE_WITH_FOLDS, index=False)

  print(f"Training data with folds saved to {config.TRAINING_FILE_WITH_FOLDS}")
