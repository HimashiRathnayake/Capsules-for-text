import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

from parameters_colab import *

def load_data():
  all_data = pd.read_csv(data_path)

  if (technique == "humor"):
    all_data = all_data[['Sentence', 'Humor']]
  elif (technique == "hate speech"):
    all_data = all_data[['Sentence', 'Hate_speech']]

  all_data.columns = ['comment', 'label']
  return all_data

def apply_oversampling(x, y):

  (unique, counts) = np.unique(y, axis=0, return_counts=True)
  print("Class Distribution Without Oversampling", counts)

  # define oversampling strategy
  if (over_sampling_technique == ""):
    return x, y
  elif (over_sampling_technique == "ROS"):
    if (technique=="humor"):
      oversample = RandomOverSampler(sampling_strategy = float(sampling_strategy))
    else:
      sampling_ratio = sampling_strategy.split(":");
      oversample = RandomOverSampler(ratio = {
          0:int(counts[0]*float(sampling_ratio[0])), 
          1:int(counts[0]*float(sampling_ratio[1])), 
          2:int(counts[0]*float(sampling_ratio[2]))
          })
  elif (over_sampling_technique == "ADASYN"):
    oversample = ADASYN(sampling_strategy="minority")
  elif (over_sampling_technique == "SMOTE"):
    oversample = SMOTE()
  elif (over_sampling_technique == "BorderlineSMOTE"):
    oversample = BorderlineSMOTE()

  # fit and apply the transform
  X_over, y_over = oversample.fit_resample(x, y)
  if(technique=="humor"):
    y_over = pd.get_dummies(y_over.flatten())

  (unique, counts) = np.unique(y_over, axis=0, return_counts=True)
  print("Class Distribution After Oversampling", counts)

  return X_over, y_over