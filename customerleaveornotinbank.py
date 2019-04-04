
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#we are read data by the pandas
dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:, 3:13].values
y=dataset.iloc[:,13].values
# Encording catgorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
ohe=OneHotEncoder(categorical_features=[1])
x=ohe.fit_transform(x).toarray()
x=x[:,1]
#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scalling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#fitting classifier to the training sets
# create your classifier here
# importing the keras library\
import keras
from keras.models import Sequential
from keras.layers import Dense

#initalising the ANN
classifier=sequential()
#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',acvtivation='relu',input_dim=''))
# adding second hidder layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu))
#add the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid')) #<-dependent variables
#compiling the ANN
classifier.compiler(optimizer='adam',loss='binary_crossentropy',metries=['accuracy'])
#fitting the Ann to the training set
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)
#predicting the test set result
y_pred=classifier.predict(x_test)
y_pred=(y_pred>.05)
# predicting the test set result
y_pred=classifier.predict(x_test)
# matrix the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusin_matrix(y_test,y_pred)











