import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("used cars and bikes.csv")
#print(dataset.columns)

sns.heatmap(dataset.isnull())
#plt.show()

plt.scatter(dataset['owner'],dataset['selling_price'])
plt.xlabel('owner')
plt.ylabel('selling_price')
#plt.show()

#visualising numerical variables
plt.figure(figsize=(15, 15))
sns.pairplot(dataset)
#plt.show()

#visualising categorical variables
plt.figure(figsize=(10, 20))
plt.subplot(4,2,1)
sns.boxplot(x = 'vehicle_type', y = 'selling_price', data = dataset)
plt.subplot(4,2,2)
sns.boxplot(x = 'fuel', y = 'selling_price', data = dataset)
plt.subplot(4,2,3)
sns.boxplot(x = 'owner', y = 'selling_price', data = dataset)
plt.tight_layout()
plt.show()

#creating dummy variables
vehicle=pd.get_dummies(dataset['vehicle_type'],drop_first=True)
#print(vehicle)
Fuel=pd.get_dummies(dataset['fuel'],drop_first=True)
#print(Fuel)
Owner=pd.get_dummies(dataset['owner'],drop_first=True)
#print(Owner)
dataset.drop(['vehicle_type','name','fuel','owner'],axis=1,inplace=True)
dataset=pd.concat([dataset,vehicle,Fuel,Owner],axis=1)
#print(dataset)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(dataset.drop('selling_price',axis=1),dataset['selling_price'],test_size=0.20,random_state=50)


from sklearn.linear_model import LinearRegression

linreg=LinearRegression()

linreg.fit(x_train,y_train)

y_pred=linreg.predict(x_test)
#print(y_pred)
y_pred1=linreg.predict([[2014,28000,1,0,0]])
#print(y_pred1)

from sklearn.metrics import mean_squared_error
#print(mean_squared_error(y_test,y_pred))

#print(linreg.score(x_test, y_test)*100,'% Prediction Accuracy')



