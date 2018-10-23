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
from matplotlib import cm

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

  processed_features["age"] = linear_scale(df["age"])
  processed_features["capital-gain"] = log_normalize(df["capital-gain"])
  processed_features["capital-loss"] = log_normalize(df["capital-loss"])
  processed_features["education-num"] = linear_scale(df["hours-per-week"])

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


def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(
        learning_rate,
        regularization_strength,
        steps,
        batch_size,
        feature_columns,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Returns:
      A `LinearClassifier` object trained on the training data.
    """

    periods = 10
    #ploting weights settings
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)

    steps_per_period = steps / periods
    sample=training_examples.join(training_targets).sample(300)
    # Create a linear regressor object.
    #my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                                          #, l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets,
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets,
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets,
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.

    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f (training)" % (period, training_root_mean_squared_error))
        print("  period %02d : %0.2f (validation)" % (period, validation_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)

        plot_weights(linear_regressor, sample, my_label='salary_bracket',
                     my_feature='hours-per-week', period=period, colors=colors)

    print("Model training finished.")


    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor


def plot_weights(model,sample,my_label, my_feature, period, colors):

    y_extents = np.array([0, sample[my_label].max()])

    weight = model.get_variable_value('linear/linear_model/%s/weights' % my_feature)[0]
    bias = model.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period])


def main():
    print("Using Tensorflow GPU")

    training_examples, training_targets_examples, test_features, test_targets = getdata()

    training_features = training_examples.tail(22162)
    training_targets = training_targets_examples.tail(22162)

    validation_features = training_examples.tail(8000)
    validation_targets = training_targets_examples.tail(8000)

    training_features_labelenc, _ = number_encode_features(training_features)
    validation_features_labelenc, _ = number_encode_features(validation_features)
    test_features_labelenc, _ = number_encode_features(test_features)

    cols = ["workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "native-country"]
    training_one_hot_enc = one_hot_encoders(training_features, cols)
    validation_one_hot_enc = one_hot_encoders(validation_features, cols)
    test_one_hot_enc = one_hot_encoders(test_features, cols)

    minimal_features = [
        "hours-per-week",
        "education-num",
        "sex"
    ]

    type=1

    if(type==1):
        minimal_training_features = training_one_hot_enc[minimal_features]
        minimal_validation_features = validation_one_hot_enc[minimal_features]
        minimal_test_features = test_one_hot_enc[minimal_features]

    elif(type==2):
        minimal_training_features = training_features_labelenc[minimal_features]
        minimal_validation_features = validation_features_labelenc[minimal_features]
        minimal_test_features = test_features_labelenc[minimal_features]
    else:
        minimal_training_features = training_features[minimal_features]
        minimal_validation_features = validation_features[minimal_features]
        minimal_test_features = test_features[minimal_features]


    classifier = train_model(
        learning_rate=0.001,
        regularization_strength=0.001,
        steps=3000,
        batch_size=100,
        feature_columns=construct_feature_columns(minimal_training_features),
        training_examples=minimal_training_features,
        training_targets=training_targets,
        validation_examples=minimal_validation_features,
        validation_targets=validation_targets)

    predict_testing_input_fn = lambda: my_input_fn(minimal_test_features,
                                                   test_targets,
                                                    num_epochs=1,
                                                    shuffle=False)

    test_predictions = classifier.predict(input_fn=predict_testing_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    test_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))

    print("Test RMSE: {}".format(test_root_mean_squared_error) )

if __name__=="__main__": main()