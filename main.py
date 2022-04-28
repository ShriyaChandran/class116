import pandas as pd
import statistics
import csv
import plotly.express as px

df= pd.read_csv("data.csv")
salary = df["EstimatedSalary"].tolist()
purchased = df["Purchased"].tolist()
print(len(salary))
fig = px.scatter(x=salary, y=purchased)
fig.show()

import plotly.graph_objects as go
age= df["Age"].tolist()
colors=[]
for i in purchased:
  if(i==1):
    colors.append("green")
  else:
    colors.append("red")

fig = go.Figure(data=go.Scatter(x= salary, y= age, mode="markers", marker= dict(color=colors)) )
fig.show()

factors = df[["EstimatedSalary", "Age"]]
purchases = df["Purchased"]
from sklearn.model_selection import train_test_split
salary_train, salary_test, purchase_train, purchase_test = train_test_split(factors, purchases, test_size = 0.25, random_state = 0 )
print(salary_train[0:10])

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
salary_train = sc_x.fit_transform(salary_train)
salary_test = sc_x.transform(salary_test)
print(salary_train[0:10])

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(salary_train, purchase_train)

purchase_pred = classifier.predict(salary_test)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(purchase_test, purchase_pred))

