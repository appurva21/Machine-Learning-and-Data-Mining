import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


data = pd.read_csv('Iris.csv')

df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X_5= df_norm.head(5)
X_5 = np.array(X_5, dtype='float32')
target = data[['Species']].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])
y_5=target.head(5)

df = pd.concat([df_norm, target], axis=1)
test2 = df.head(5)
train_test_per = 60/100.0
df['train'] = np.random.rand(len(df)) < train_test_per

train = df[df.train == 1]
train = train.drop('train', axis=1).sample(frac=1)

test = df[df.train == 0]
test = test.drop('train', axis=1)

X = train.values[:,:4]

targets = [[1,0,0],[0,1,0],[0,0,1]]
y_5 = np.array([targets[int(x)] for x in y_5.values[:,:1]])
y = np.array([targets[int(x)] for x in train.values[:,4:5]])

num_inputs = len(X[0])
hidden_layer_neurons = 5
np.random.seed(4)
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1

num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1

learning_rate = 0.3

print('\nNo. of Inputs: ',num_inputs) 
print('No. Of hidden layers: 1')
print('No. Of nodes in hidden layer: ',hidden_layer_neurons)
print('Learning Rate: ',learning_rate)

max_iterations=50000
epoch=0
prev_er=0
while True:
	
    l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    er = (abs(y - l2)).mean()
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += X.T.dot(l1_delta) * learning_rate
    if ((epoch!=0) and (er<0.05) and (abs(prev_er-er)<0.0001)):
        epoch+=1
        print()
        print('Status: Converged')
        break
    if epoch>=max_iterations:
        print('Status: Diverged')
        break
    prev_er=er
    epoch=epoch+1
print('\nCost Value:', er)
print('Number Of Iterations: ',epoch)

X = test.values[:,:4]
y = np.array([targets[int(x)] for x in test.values[:,4:5]])

l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

np.round(l2,3)

yp = np.argmax(l2, axis=1)
res = yp == np.argmax(y, axis=1)
correct = np.sum(res)/len(res)

testres = test[['Species']].replace([0,1,2], ['Iris-setosa','Iris-versicolor','Iris-virginica'])

testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1,2], ['Iris-setosa','Iris-versicolor','Iris-virginica'])

#print(testres)
print('Result for 40% test data:',sum(res),'/',len(res), ':','Accuracy', (correct*100),'%\n')


#predicting output for first 5 examples
l1 = 1/(1 + np.exp(-(np.dot(X_5, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

np.round(l2,3)

yp = np.argmax(l2, axis=1)
res = yp == np.argmax(y_5, axis=1)
correct = np.sum(res)/len(res)

testres = test2[['Species']].replace([0,1,2], ['Iris-setosa','Iris-versicolor','Iris-virginica'])

testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1,2], ['Iris-setosa','Iris-versicolor','Iris-virginica'])
print('Prediction result for First 5 input data:\n',testres)
