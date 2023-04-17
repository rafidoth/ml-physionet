from mydatasets import load_data
from pyspark import SparkContext
from pyspark.sql import SparkSession

num_samples=12

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# Set a correct path to the seizure data file you downloaded
PATH_TRAIN_FILE = "../data/training/"
PATH_VALID_FILE = "../data/testing/"
PATH_TEST_FILE = "../data/vaildation/"

Xtrain, ytrain = load_data(PATH_TRAIN_FILE, num_samples)
Xvalid, yvalid = load_data(PATH_VALID_FILE, num_samples)
Xtest, ytest = load_data(PATH_TEST_FILE, num_samples)

from systemml.mllearn import SVM

svm = SVM(spark, fit_intercept=True, max_iter=100, tol=0.0001, C=0.2, is_multi_class=True)

y_pred = svm.fit(Xtrain, ytrain).predict(Xtest)

from sklearn.metrics import classification_report, accuracy_score

target_names = ['sleep_stage_1', 'sleep_stage_2', 'sleep_stage_3', 'sleep_stage_4', 'sleep_stage_5']
print(classification_report(y_pred, ytest, target_names=target_names))
print('crude accuracy: ', accuracy_score(y_pred, ytest))

