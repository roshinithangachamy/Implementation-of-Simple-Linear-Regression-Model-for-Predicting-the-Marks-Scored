# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored


## AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Start the program.
2.Import the standard Libraries like numpy,pandas,matplotlib and sklearn for handling data.
3.Read the dataset that contains features and target variables.
4.Divide the dataset into training and test datasets and fit the linear regression model.
5.Assign the points for representing in the graph.
6.Predict the regression for marks by using the representation of the graph.
7.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: T.Roshini
RegisterNumber: 212223230175

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()
df.tail()
# segregating data to variables
X=df.iloc[:,:-1].values
print("X:",X)
Y=df.iloc[:,1].values
print("Y:",Y)
# splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
# displaying predicted values
Y_pred
print("Y_pred:",Y_pred)
Y_test
print("Y_test:",Y_test)
# graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse) 

```

## Output:

#### HEAD:

![image](https://github.com/user-attachments/assets/e69d04be-18df-4c63-8b87-9581b51c6f4d)

#### TAIL:

![image](https://github.com/user-attachments/assets/91c47099-db08-4665-b763-9658e74037d5)

#### X:

![image](https://github.com/user-attachments/assets/412d8f40-51eb-4d1e-81aa-be409fa3149c)

#### Y:

![image](https://github.com/user-attachments/assets/2da88b09-0cbf-441a-b061-b62fedd38089)

#### Y_PRED:

![image](https://github.com/user-attachments/assets/9f4d36d3-4de2-416d-8ca7-978fb8392bba)

#### Y_TEST:

![image](https://github.com/user-attachments/assets/2931b3c5-22de-44e6-8bcb-b51df08c8673)

#### TRAINING SET:

![download](https://github.com/user-attachments/assets/2dedabd9-4458-4649-8ab2-0e6bcced3edf)

#### TEST SET:

![download](https://github.com/user-attachments/assets/31467f20-2c4c-4780-adfd-f45f64cffe82)

#### MSE:

![image](https://github.com/user-attachments/assets/eae1218f-801d-43cd-aed9-388952d57fba)

#### RMSE:

![image](https://github.com/user-attachments/assets/401d7762-92b4-42e2-aa63-71b7cb62c5d7)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
