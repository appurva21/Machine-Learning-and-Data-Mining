#implementing logistic regression library function

# Required Python Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

DATA_SET_PATH = 'diabetes.csv'

def train_logistic_regression(train_x, train_y):
	logistic_regression_model = LogisticRegression()
	logistic_regression_model.fit(train_x, train_y)
	return logistic_regression_model

def model_accuracy(trained_model, features, targets):
	accuracy_score = trained_model.score(features, targets)
	return accuracy_score
	
def main():

	# Load the data set for training and testing the logistic regression classifier
	dataset = pd.read_csv(DATA_SET_PATH, names = ['x1','x2','x3','x4','x5','x6','x7','x8','y'], skiprows=1)

	training_features = ['x1','x2','x3','x4','x5','x6','x7','x8']
	target = 'y'

	# Train , Test data split
	train_x, test_x, train_y, test_y = train_test_split(dataset[training_features], dataset[target], test_size=0.5)

	# Training Logistic regression model
	trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
		
	train_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)
	
	print()
	print("Train Accuracy for 50% of data given:: ", train_accuracy*100)
	print("Output for given test cases:: ",trained_logistic_regression_model.predict([ [5,153,76,40,120,36.1,0.471,26],[4,90,70,30,0,35.7,0.27,53],[5,54,70,34,0,30.8,0.090,29]]))
	print()
if __name__ == "__main__":
	main()