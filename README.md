# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Start the program.

2.Import the standard Libraries.

3.Set variables for assigning dataset values.

4.Import linear regression from sklearn.

5.Assign the points for representing in the graph.

6.Predict the regression for marks by using the representation of the graph.

7.Compare the graphs and hence we obtained the linear regression for the given datas.

8.Stop te program.

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
X

Y=df.iloc[:,1].values
Y

# splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

# displaying predicted values
Y_pred

Y_test

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

![Screenshot 2024-08-24 090451](https://github.com/user-attachments/assets/dbe7969d-33e6-47a2-8cf1-db842272c37b)

![Screenshot 2024-08-24 090503](https://github.com/user-attachments/assets/86efaf97-5759-45c1-8cdc-54e768946ca0)

![Screenshot 2024-08-24 090516](https://github.com/user-attachments/assets/56357c81-25a4-4dd7-9110-977c9f266433)

![Screenshot 2024-08-24 090527](https://github.com/user-attachments/assets/f2353e75-4eeb-4fc5-9bd7-b1a74f2e8fca)

![Screenshot 2024-08-24 090544](https://github.com/user-attachments/assets/c009ac73-8218-442d-8c96-c966e6db7e05)

![Screenshot 2024-08-24 090634](https://github.com/user-attachments/assets/15f80ceb-ad49-4eff-be30-d55cd88c08d7)

![Screenshot 2024-08-24 090643](https://github.com/user-attachments/assets/1ec6b822-0ece-4f24-9fe1-0803d6937729)

![Screenshot 2024-08-24 090654](https://github.com/user-attachments/assets/cbbf89cc-c815-4ce1-97ad-1c96b782c83d)

![Screenshot 2024-08-24 090703](https://github.com/user-attachments/assets/f84d05fe-7b58-4c48-b8c2-45fbfe4c8010)

![Screenshot 2024-08-24 090710](https://github.com/user-attachments/assets/6e05218e-beb2-4935-a148-6056a7837635)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
