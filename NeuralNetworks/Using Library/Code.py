#Import required libraries 
import keras 
import pandas as pd 
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold

#Neural network module
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 

# Make results reproducible
seed = 1234
np.random.seed(seed)


# Loading the dataset
dataset = pd.read_csv('Iris.csv')
dataset = pd.get_dummies(dataset, columns=['Species']) # One Hot Encoding
values = list(dataset.columns.values)
y = dataset[values[-3:]]
y_5 = y.head(5)

y = np.array(y, dtype='float32')
y_5 = np.array(y_5, dtype='float32')

X = dataset[values[1:-3]]
X_5 = X.head(5)
X = np.array(X, dtype='float32')
X_5 = np.array(X_5, dtype='float32')


# Shuffle Data
indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]

#normalization
#X_normalized=normalize(X_values,axis=0)
X_normalized= X_values

# Creating a Train and a Test Dataset
test_size = int(0.2*len(dataset))
X_test = X_normalized[-test_size:]
X_train = X_normalized[:-test_size]
y_test = y_values[-test_size:]
y_train = y_values[:-test_size]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

#Defining Model
model = Sequential()
model.add(Dense(4,input_dim=4,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dropout(0.2))#to prevent problem of over-fitting
model.add(Dense(3,activation='sigmoid'))

#Compiling Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Training Model
#model.fit(X_train,y_train,batch_size=10,epochs=300, verbose=0)
model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=100,verbose=0)

# evaluate the model for test data
scores = model.evaluate(X_train,y_train)
print("\nTraining Accuracy",scores[1]*100)

# evaluate the model for test data
scores = model.evaluate(X_test,y_test)
print("\nTest Data Accuracy",scores[1]*100)

#Output For First Five examples from dataset
#[1,0,0]-->Iris-setosa
#[0,1,0]-->Iris-versicolor
#[0,0,1]-->Iris-virginica
print("\nOutput For First Five examples from dataset")
predictions=model.predict(X_5)
predictions=predictions.round()
print("INDEX:\n#[1,0,0]-->Iris-setosa\n#[0,1,0]-->Iris-versicolor\n#[0,0,1]-->Iris-virginica\n")
print(predictions)
