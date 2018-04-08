#Code in Python3
#imports
import numpy as np
import pandas as pd
import csv
from collections import Counter

#list for sample data
samples=[]

#reading data from excel sheet
with open('boston_housing.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in reader:
			if Counter(row[0])!=Counter('CRIM'):
				temp = [float(i) for i in row]
				samples.append(temp)
			
trainingData = pd.DataFrame(samples, columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','y'])
trainingData.insert(0, 'x0', np.ones(506))

#calculating theta using Normal Equation
X = trainingData[['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']]
y = trainingData[['y']]
X_transpose = X.T
X_transpose_dot_X = X_transpose.dot(X)
X_transpose_dot_X_inverse = np.linalg.inv(X_transpose_dot_X)
X_transpose_dot_X_inverse_dot_X_transpose = X_transpose_dot_X_inverse.dot(X.T)
theta = X_transpose_dot_X_inverse_dot_X_transpose.dot(y)
print("Theta")
print(theta) #value of theta

#calculating value of cost function
predictedY=[]
sum_of_sqaure = 0
for row in samples:
	hTheta=0
	for i in range(1,14,1):
		hTheta = hTheta + theta[i]*row[i-1]
	hTheta = hTheta + theta[0]
	predictedY.append(hTheta)
	sum_of_sqaure = sum_of_sqaure + ((hTheta-row[13])*(hTheta-row[13]))
cost_value = sum_of_sqaure/(2*506)
print("Value of cost function")
print(cost_value) # value of cost function

#test data
test = [0.085,13.0,10.5,1.0,0.8,4.78,39.0,5.5,5.5,331.0,13.3,390.5,17.71]
#output for test data
output = 0
for i in range(1,14,1):
	output = output + theta[i]*test[i-1]
output = output + theta[0]
print("Output for the test data")
print(output)#Output for the test data = 15.13
