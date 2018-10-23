import math
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from tensorflow.python.data import Dataset
from time import time
from math import sqrt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#--
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#functions
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

  series = pd.Series(processed_features['sex']).astype(str)
  processed_features['sex'] = series.apply(lambda x: (1 if x == "Male" else 2))
  # Create a synthetic feature.
  return processed_features

def normalize(df):
    """Returns a version of the input `DataFrame` that has all its features normalized."""
    processed_features = pd.DataFrame()

    processed_features["latitude"] = linear_scale(df["latitude"])
    processed_features["longitude"] = linear_scale(df["longitude"])

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

  series = df["salary_bracket"].astype(str)
  output_targets["salary_bracket"] = series.apply(lambda x:(1 if x == "<=50K" else 2))

  output_targets.astype(float)
  #print(output_targets["salary_bracket"].value_counts())

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

def one_hot_encoders(df, cols):
    result = df.copy()
    for col in cols:
        result[col] = pd.get_dummies(df[col]).values.tolist()

    return result


#read in data
print("Using Tensorflow GPU")

training_examples, training_targets_examples, test_features, test_targets = getdata()

training_features = training_examples.tail(22162)
training_targets = training_targets_examples.tail(22162)

validation_features = training_examples.tail(8000)
validation_targets = training_targets_examples.tail(8000)

minimal_features = [
    "hours-per-week",
    "education-num",
    "sex"
]

minimal_training_features = training_features[minimal_features]
minimal_validation_features = validation_features[minimal_features]
minimal_test_features = test_features[minimal_features]


# Create linear regression object
regr = linear_model.LinearRegression(normalize=True)
#print(training_data.describe())
#print(test_examples.describe())
print("Fitting on the data...")
# Train the model using the training sets
regr.fit(minimal_training_features, training_targets)

# The coefficients
print('Coefficients: \n', regr.coef_)

print("Making predictions...")

pred_train = regr.predict(minimal_training_features)

print("Root mean squared error on traning data: %.2f"
      % sqrt(mean_squared_error(training_targets, pred_train)))

# The mean squared error

# Make predictions using the testing set
pred_test = regr.predict(minimal_test_features)

print("Root mean squared error on test data: %.2f"
      % sqrt(mean_squared_error(test_targets, pred_test)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_targets, pred_test))

#validation

pred_val = regr.predict(minimal_validation_features)

print("Root mean squared error validation: %.2f"
      % sqrt(mean_squared_error(validation_targets, pred_val)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f validation' % r2_score(validation_targets, pred_val))






'''
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(np.concatenate(pred_test))
calibration_data["targets"] = pd.Series(np.concatenate(test_targets))
print(calibration_data.describe())
print(calibration_data)

# Plot outputs
plt.scatter(pred, test_targets,  color='black')
#plt.plot(test_examples, pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


#using SGD
clf = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=5,
       n_jobs=3, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)

clf.fit(my_feature_train, targets_train)

pred = clf.predict(my_feature_test)


# The coefficients
print('Coefficients: \n', clf.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(targets_test, pred))

print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(targets_test, pred)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(targets_test, pred))

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(pred)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data.describe())
print(calibration_data)

# Plot outputs
plt.scatter(my_feature_test, targets_test,  color='black')
plt.plot(my_feature_test, pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
'''