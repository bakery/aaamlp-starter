import os
import argparse
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

import config
import model_dispatcher

def run(fold, model):
  df = pd.read_csv(config.TRAINING_FILE_WITH_FOLDS)

  # train with everything that is not in the current fold
  df_train = df[df.kfold != fold].reset_index(drop=True)

  # validate on everything in the current fold
  df_valid = df[df.kfold == fold].reset_index(drop=True)

  # drop the **label** col from dataframe and convert it to
  # a numpy array
  # target is label column in the dataframe
  x_train = df_train.drop("label", axis=1).values
  y_train = df_train.label.values

  # same for validation
  x_valid = df_valid.drop("label", axis=1).values
  y_valid = df_valid.label.values

  clf = model_dispatcher.models[model]

  clf.fit(x_train, y_train)

  preds = clf.predict(x_valid)

  accuracy = metrics.accuracy_score(y_valid, preds)

  print(f"Fold={fold}, Accuracy={accuracy}")

  joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--fold", type=int, required=True)
  parser.add_argument("--model", type=str, choices=model_dispatcher.models.keys(), required=True)

  args = parser.parse_args()
  
  run(fold=args.fold, model=args.model)
