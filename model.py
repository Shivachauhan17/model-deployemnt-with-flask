import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

dataset=pd.read_csv('C:/Users/shiva/Desktop/codes/python/deployement/Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.2,random_state=0)
model=LinearRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
pickle.dump(model,open('model.pkl','wb'))
