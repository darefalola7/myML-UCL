from __future__ import print_function

import math
import os
#from IPython import display
from matplotlib.pyplot import show
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import Imputer
import missingno as msno
import math as mt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
config = tf.ConfigProto()
session = tf.Session(config=config)
config.gpu_options.allow_growth = True

def preprocess_features(df):
  """Prepares input features from California housing data set.

  Args:
    census income_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = df[
    ["age",
     "workclass",
     "education",
     "education-num",
     "marital-status",
     "occupation",
     "relationship",
     "race",
     "sex",
     "capital-gain",
     "capital-loss",
     "hours-per-week",
     "native-country"]]

  processed_features = selected_features.copy()

  #normalise features that can be normalized
  processed_features["age"] = linear_scale(df["age"])
  processed_features["capital-gain"] = log_normalize(df["capital-gain"])
  processed_features["capital-loss"] = log_normalize(df["capital-loss"])
  processed_features["hours-per-week"] = linear_scale(df["hours-per-week"])

  series = pd.Series(processed_features['sex']).astype(str)
  processed_features['sex'] = series.apply(lambda x: (1 if x == "Male" else 2))
  # Create a synthetic feature.
  return processed_features

def normalize(df):
    """Returns a version of the input `DataFrame` that has all its features normalized."""

    processed_features = pd.DataFrame()

    processed_features["age"] = linear_scale(df["age"])
    processed_features["education-num"] = linear_scale(df["education-num"])
    processed_features["education-num"] = linear_scale(df["hours-per-week"])

    return processed_features

def preprocess_targets(df):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    census income_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  #salary = pd.Series(df["salary_bracket"].values)
  # Scale the target to be in units of thousands of dollars.

  output_targets["salary_bracket"] = df["salary_bracket"].apply(lambda x:(1 if set(x.split()) == set("<=50K".split()) else 2))
#  output_targets["salary_bracket"] = df["salary_bracket"]
#  output_targets.astype('str')
  return output_targets

def getdata():
    cols=["age",
     "workclass",
     "fnlwgt",
     "education",
     "education-num",
     "marital-status",
     "occupation",
     "relationship",
     "race",
     "sex",
     "capital-gain",
     "capital-loss",
     "hours-per-week",
     "native-country",
     "salary_bracket"]

    training = pd.read_csv("adult.data", sep=r'\s*,\s*', names=cols, na_values="?", engine='python')
    test = pd.read_csv("adult.test",sep=r'\s*,\s*', names=cols, na_values="?", engine='python')

    training = training.dropna()
    test = test.dropna()


    training = training.reindex(np.random.permutation(training.index))
    training_features = preprocess_features(training)
    training_targets = preprocess_targets(training)

    test_features = preprocess_features(test)
    test_targets = preprocess_targets(test)

    return training_features, training_targets, test_features, test_targets

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def log_normalize(series):
  return series.apply(lambda x:math.log(x+1.0))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(
    min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))

def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column].astype(str))
    return result, encoders

def one_hot_encoders(df, cols):
    result = df.copy()
    for col in cols:
        result[col] = pd.get_dummies(df[col]).values.tolist()

    return result

def impute_nas(df, col):
    imputer = Imputer(missing_values=np.nan, strategy='median', axis=0)
    df[col] = imputer.fit_transform(df[col])





def main():
    print("Using Tensorflow GPU")

    training_features, training_targets, test_features, test_targets = getdata()

    '''
    print("training feature: ")
    print(training_features.head())
    print(training_features.describe())
    print(training_targets.describe())
    print(training_features.info())
    print("Nas", training_features.isnull().sum())
    


    print("test feature: ")
    print(test_features.head())
    print(test_features.describe())
    print(test_targets.describe())
    print(test_features.info())
    #number_encode_features(test_features)
    
    '''
    #msno.matrix(training_features.sample(500))

    #plt.show()
    training_features_labelenc, _ = number_encode_features(training_features)
    test_features_labelenc, _ = number_encode_features(test_features)

    cols = ["workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "native-country"]
    print("Only categorical")
    training_one_hot_enc = one_hot_encoders(training_features, cols)
    test_one_hot_enc = one_hot_encoders(test_features, cols)

    training_features_labelenc['Salary'] = training_targets

    others = [
        "age",
        "sex",
        'education-num',
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        'Age_Hpw',
        'Age_gain',
        'Hpw_Hpw',
        'Salary'
    ]

    training_features_labelenc['Age_Hpw'] = training_features_labelenc['age'] * training_features_labelenc['hours-per-week']
    training_features_labelenc['Age_gain'] = training_features_labelenc['age'] * training_features_labelenc[
        'capital-gain']
    training_features_labelenc['Hpw_Hpw'] = training_features_labelenc['hours-per-week'] * training_features_labelenc[
        'hours-per-week']

    sns.heatmap(training_features_labelenc[others].corr(), square=True, annot=True)


    plt.show()
'''
    fig = plt.figure(figsize=(17, 15))
    cols = 4
    rows = mt.ceil(float(training_features.shape[1]) / cols)
    for i, column in enumerate(training_features.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if training_features.dtypes[column] == np.object:
            training_features[column].value_counts().plot(kind="bar", axes=ax)
        else:
            training_features[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=1.2, wspace=0.2)
    
'''

if __name__=="__main__": main()
