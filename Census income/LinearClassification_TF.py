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
  processed_features["education-num"] = linear_scale(df["hours-per-week"])

  series = pd.Series(processed_features['sex']).astype(str)
  processed_features['sex'] = series.apply(lambda x: (1 if x == "Male" else 2))
  # Create a synthetic feature.
  return processed_features


def preprocess_targets(df):
  #Prepares target features (i.e., labels) from California housing data set.

  output_targets = pd.DataFrame()

  series = df["salary_bracket"].astype(str)
  output_targets["salary_bracket"] = series.apply(lambda x:(0.0 if x == "<=50K" else 1.0))

  output_targets.astype(float)

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

#use only for values very close to zero and above! Negative values will give you error
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


def construct_feature_columns(input_features, basic):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def construct_feature_columns(input_features):

    #num_fts = [tf.feature_column.numeric_column(my_feature) for my_feature in input_features[others]]
    numerical_age = tf.feature_column.numeric_column("age")
    numerical_sex = tf.feature_column.numeric_column("sex")
    numerical_capital_gain = tf.feature_column.numeric_column("capital-gain")
    numerical_capital_loss = tf.feature_column.numeric_column("capital-loss")
    numerical_hours_per_week = tf.feature_column.numeric_column("hours-per-week")

    categorical_workclass = tf.feature_column.categorical_column_with_identity('workclass', num_buckets=max(
        input_features['workclass'])+1)
    categorical_education = tf.feature_column.categorical_column_with_identity('education', num_buckets=max(
        input_features['education'])+1)
    categorical_marital_status = tf.feature_column.categorical_column_with_identity('marital-status', num_buckets=max(
        input_features['marital-status'])+1)
    categorical_occupation = tf.feature_column.categorical_column_with_identity('occupation', num_buckets=max(
        input_features['occupation'])+1)
    categorical_relationship = tf.feature_column.categorical_column_with_identity('relationship', num_buckets=max(
        input_features['relationship'])+1)
    categorical_race = tf.feature_column.categorical_column_with_identity('race', num_buckets=max(
        input_features['race'])+1)
    categorical_native_country = tf.feature_column.categorical_column_with_identity('native-country', num_buckets=max(
        input_features['native-country'])+1)

    feature_columns =set([
        numerical_age,
        numerical_sex,
        numerical_capital_gain,
        numerical_capital_loss,
        numerical_hours_per_week,
        categorical_workclass,
        categorical_education,
        categorical_marital_status,
        categorical_occupation,
        categorical_relationship,
        categorical_race,
        categorical_native_country
    ])


    return feature_columns

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
    as well as a plot of the training and validation loss over time

    Returns:
      A `LinearClassifier` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    #my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # Create a linear classifier object.
    #my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
     #                                     l1_regularization_strength=regularization_strength)
    my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=my_optimizer)

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
    print("Log loss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )

        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("(Validation)  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)

    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()
    plt.show()

    return linear_classifier

'''
def plot_weights(model,sample,my_label, my_feature, period):
    y_extents = np.array([0, sample[my_label].max()])

    weight = model.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = model.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period])
'''

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

    others = [
        "age",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]

    training_one_hot_enc = one_hot_encoders(training_features, cols)
    validation_one_hot_enc = one_hot_encoders(validation_features, cols)
    test_one_hot_enc = one_hot_encoders(test_features, cols)

    result = pd.concat([training_one_hot_enc, training_features[others]], axis=1, sort=False)



    minimal_features = [
        "age",
        "education",
         "workclass",
         "marital-status",
         "occupation",
         "relationship",
         "race",
         "sex",
         "capital-gain",
         "capital-loss",
         "hours-per-week",
         "native-country"
    ]

    type=2

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
        steps=4000,
        batch_size=300,
        feature_columns=construct_feature_columns(minimal_training_features),
        training_examples=minimal_training_features,
        training_targets=training_targets,
        validation_examples=minimal_validation_features,
        validation_targets=validation_targets)

    predict_testing_input_fn = lambda: my_input_fn(minimal_test_features,
                                                   test_targets,
                                                    num_epochs=1,
                                                    shuffle=False)

    test_probabilities = classifier.predict(input_fn=predict_testing_input_fn)
    test_probabilities =np.array([item['probabilities'] for item in test_probabilities])

    print(test_targets["salary_bracket"].value_counts())
    test__error = metrics.log_loss(test_targets["salary_bracket"], test_probabilities)

    print("Test Loss: {}".format(test__error))

    evaluation_metrics = classifier.evaluate(input_fn=predict_testing_input_fn)
    print("AUC on the test set: %0.2f" % evaluation_metrics['auc'])
    print("Accuracy on the test set: %0.2f" % evaluation_metrics['accuracy'])

    for k in evaluation_metrics.keys():
        print(k, evaluation_metrics[k])

    test_probabilities = classifier.predict(input_fn=predict_testing_input_fn)
    # Get just the probabilities for the positive class.
    test_probabilities = np.array([item['probabilities'][1] for item in test_probabilities])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
        test_targets, test_probabilities)
    plt.plot(false_positive_rate, true_positive_rate, label="our model")
    plt.plot([0, 1], [0, 1], label="random classifier")
    _ = plt.legend(loc=2)
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.show()
    #print("Test pred:", test_probabilities)

if __name__=="__main__": main()