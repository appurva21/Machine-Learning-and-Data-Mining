
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sigmoid function
def sigmoid(X):
	return 1/(1 + np.exp(-X))
	
#function for calculating cost value for given theta
def cost_function(X,y,theta):
	m = len(y)
	hypothesis = sigmoid(X.dot(theta))
	J = (-y)*(np.log(hypothesis)) - (1-y)*((np.log(1 - hypothesis)))
	return J.mean()

#function for finding theta using gradient_descent technique for given learning rate alpha
def gradient_descent(X,y,theta,alpha,iterations,max_iterations):
	global cost
	prev_cost=cost
	m = len(y)
	while True:
		hypothesis = sigmoid(X.dot(theta))
		error = hypothesis - y
		gradient = (X.T.dot(error))/m
		theta = theta - alpha*gradient
		cost = cost_function(X,y,theta)
		if ((iterations!=0) and (abs(prev_cost-cost)<0.00001)) :
			iterations+=1
			print()
			print('Status: Converged')
			break
		if iterations>=max_iterations:
			print()
			print('Status: Diverged')
			print()
			break
		prev_cost = cost
		iterations+=1
	return theta,iterations

#Predicting output for test data
def predict(testData,theta):
	m = len(testData)
	testData = np.array(testData)
	hypothesis = sigmoid(testData.dot(theta))
	p = 1*(hypothesis>=0.5)
	return p,hypothesis

	
#reading data from excel sheet
data = pd.read_csv('diabetes.csv', names = ['x1','x2','x3','x4','x5','x6','x7','x8','y'], skiprows=1)

#normalizing data as (data-mean/range)
avg_df = data.mean(axis=0)
max_df = data.max(axis=0)
min_df = data.min(axis=0)
norm_data_df = (data - avg_df)/(max_df-min_df)

#X->features Y->class attribute
X_all = norm_data_df[['x1','x2','x3','x4','x5','x6','x7','x8']]
y_all = data[['y']]
m = len(y_all)
X=X_all[:int(m/2)]
y=y_all[:int(m/2)]
X.insert(0, 'x0', np.ones(int(m/2)))


#initializations
iterations = 0
max_iterations = 300000 #for checking divergence
alpha = 1.8
X = np.array(X)
y = np.array(y).flatten()
theta = [0] * 9


#initial cost value	
cost = cost_function(X,y,theta)

#Result
(theta,iterations) = gradient_descent(X,y,theta,alpha,iterations, max_iterations)
# when converged
if (iterations<max_iterations):
	print()
	print('Learning Rate:: ',alpha)
	print('Number of Iterations:: ', iterations)
	print()
	print('Value of cost function:: ',cost)
	print()
	print('Learned Weight Vector:: ',theta)
	print()

	#Testing the classifier
	
	testData = pd.DataFrame([ [5,153,76,40,120,36.1,0.471,26],[4,90,70,30,0,35.7,0.27,53],[5,54,70,34,0,30.8,0.090,29]],columns=['x1','x2','x3','x4','x5','x6','x7','x8'])
	testData_norm = (testData - avg_df)/(max_df-min_df)
	testData_norm = testData_norm[['x1','x2','x3','x4','x5','x6','x7','x8','y']]
	testData_norm = testData_norm.drop(labels='y',axis=1)
	testData_norm.insert(0, 'x0', np.ones(len(testData)))
	p,hypothesis = predict(testData_norm,theta)
	print("Output For Given Test Cases:: ",p)

	testData_norm = X_all[int(m/2):]
	testData_norm.insert(0, 'x0', np.ones(int(m/2)))

	p,hypothesis = predict(testData_norm,theta)

	#calculating accuracy for learned classifier
	y_test = y_all[int(m/2):]
	y_test = np.array(y_test).flatten()
	print ('Train Accuracy for 50% of data given:: ',((y_test[np.where(p == y_test)].size / y_test.size) * 100.0));