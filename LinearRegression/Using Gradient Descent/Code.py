
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading data from excel sheet
data = pd.read_csv('boston_housing.csv', names = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','y'], skiprows=1)

#normalizing data as (data-mean/range)
avg_df = data.mean(axis=0)
max_df = data.max(axis=0)
min_df = data.min(axis=0)
norm_data_df = (data - avg_df)/(max_df-min_df)


#X->features Y->class attribute(MEDV)
X = norm_data_df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']]
y = data[['y']]
m = len(y)
X.insert(0, 'x0', np.ones(m))


#initializations
iterations = 0
max_iterations = 30000 #for checking divergence
alpha = 1.9
X = np.array(X)
y = np.array(y).flatten()
theta = [0] * 14

#function for calculating cost value for given theta
def cost_function(X,y,theta):
	m = len(y)
	J = np.sum((X.dot(theta) - y)*(X.dot(theta) - y))/(2*m)
	return J
	
#function for finding theta using gradient_descent technique for given learning rate alpha
def gradient_descent(X,y,theta,alpha,iterations):
	global cost
	prev_cost=cost
	m = len(y)
	while True :
		hypothesis = X.dot(theta)
		error = hypothesis - y
		gradient = X.T.dot(error)/m
		theta = theta - alpha*gradient
		cost = cost_function(X,y,theta)		
		if ((iterations!=0) and (abs(prev_cost-cost)<0.00001)) :
			iterations+=1
			print()
			print('Status: Converged')
			break
		if iterations>=max_iterations:
			print('Status: Diverged')
			break
		prev_cost = cost
		iterations+=1
	return theta,iterations
	
#intial cost value	
cost = cost_function(X,y,theta)

#Result
(theta,iterations) = gradient_descent(X,y,theta,alpha,iterations)
print()
print('Learning Rate: ',alpha)
print('Number of Iterations: ', iterations)
print()
print('Value of cost function: ',cost)
print()

print('Learned Weight Vector: ',theta)
print()
print('Value of cost function: ',cost)
print()

#Calculating output for test data
def housePrice(testData,theta,avg_df,max_df,min_df,num):
	testData_norm = (testData - avg_df)/(max_df-min_df)
	testData_norm = testData_norm[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','y']]
	testData_norm = testData_norm.drop(labels='y',axis=1)
	testData_norm.insert(0, 'x0', np.ones(1))
	testData_norm = np.array(testData_norm)
	ans = np.sum(testData_norm.dot(theta))
	print('Output for TEST DATA',num,'-->',ans)
	
#TEST DATA 
testData1 =  pd.DataFrame([[0.0101, 30, 5.19, 0, 0.0493, 6.059,37.3, 4.8122,1, 430,19.6, 375.21, 8.51 ]],columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13'])
testData2 =  pd.DataFrame([[0.02501, 35, 4.15, 1, 0.77, 8.78, 81.3, 2.5051, 24, 666, 17, 382.8, 11.48]],columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13'])
testData3 =  pd.DataFrame([[3.67822, 0, 18.1, 1, 0.7, 6.649, 98.8, 1.1742, 24, 711, 20.2, 398.28, 18.07]],columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13'])

housePrice(testData1,theta,avg_df,max_df,min_df,1)
housePrice(testData2,theta,avg_df,max_df,min_df,2)
housePrice(testData3,theta,avg_df,max_df,min_df,3)




