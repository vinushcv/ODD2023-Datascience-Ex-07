# Ex-07-Feature-Selection
## AIM:
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation:
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM:
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE AND OUTPUT:
  Name: vinush.cv
  
  Reg no:212222230176

```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/titanic_dataset.csv")

df.columns
```
<img width="293" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/502e903e-18ba-4045-b25c-6b7f7cdb3c0f">


```python
df.shape
```
<img width="47" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/a433d228-040d-40c3-ba92-b2d4b7a58c04">

```python
x=df.drop("Survived",1)
y=df['Survived']
```
<img width="647" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/93831ad4-8b61-465c-aa1f-17c8767d447d">


```python
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)

df1.columns
```
<img width="371" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/ef53fad2-5734-477c-8d87-4e7883f8e066">

```python
df1['Age'].isnull().sum()
```
<img width="28" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/04e7ffb3-2fa4-49fd-ae74-cacdd510db6c">

```python
df1['Age'].fillna(method='ffill')
```
<img width="163" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/39519834-9b1b-4db4-bdae-e32e4a152e46">

```python
df1['Age']=df1['Age'].fillna(method='ffill')

df1['Age'].isnull().sum()
```
<img width="22" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/1ccdfdb0-6160-45d5-acbc-6436d0f496a4">

```python
feature=SelectKBest(mutual_info_classif,k=3)

df1.columns
```
<img width="370" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/6daa9312-5ee2-4866-a8dd-7502d48260e5">

```python
cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]

df1.columns
```
<img width="367" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/b4b24e0e-afd7-4ad4-9ea9-212018af4125">

```python
x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns
```
<img width="339" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/7c9f9df5-162f-4bb1-93d7-985385e16bbd">

```python
y=y.to_frame()

y.columns
```
<img width="131" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/af944d3a-1a38-4ef0-9359-c83e99a7ecd0">

```python
from sklearn.feature_selection import SelectKBest

data=pd.read_csv("/content/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']

x
```
<img width="301" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/da6ba24e-4d17-41d5-bb69-e5bcec212be5">

```python
data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data[ "Embarked" ]=data ["Embarked"] .astype ("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data[ "Embarked" ]=data ["Embarked"] .cat.codes

data
```
<img width="538" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/56fb8837-542e-4b5a-8420-f3dc51ca9c69">

```python
k=5
selector = SelectKBest(score_func=chi2,k=k)
x_new = selector.fit_transform(x,y)

selected_feature_indices = selector.get_support(indices=True)

selected_feature_indices = selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features: ")
print(selected_features)
```
<img width="277" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/d907d91f-a21a-4896-bdde-4208d14f80de">

```python
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

sfm = SelectFromModel(model, threshold='mean')

sfm.fit(x,y)
```
<img width="220" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/4146e1ff-6036-4e62-bfef-6e71ee860624">

```python
selected_feature = x.columns[sfm.get_support()]

print("Selected Features:")
print(selected_feature)
```
<img width="413" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/5cf4f92b-d7d4-4011-ace4-f7b6020353ec">

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()

num_features_to_remove =2
rfe = RFE(model, n_features_to_select=(len(x.columns) - num_features_to_remove))

rfe.fit(x,y)
```
<img width="405" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/1963670f-72d6-47c2-b70f-3ae51b9a4d50">

```python
selected_features = x.columns[rfe.support_]

print("Selected Features:")
print(selected_feature)
```
<img width="210" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/96eb3507-2f82-458a-b746-47c812633e62">

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x,y)
```

<img width="163" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/04be9a03-add3-4559-a981-29c19e27a09f">

```python
feature_importances = model.feature_importances_

threshold = 0.15

selected_features = x.columns[feature_importances > threshold]

print("Selected Features:")
print(selected_feature)
```

<img width="280" alt="image" src="https://github.com/TejaswiniGugananthan/ODD2023-Datascience-Ex-07/assets/121222763/9ed6ba84-ac85-448f-8bae-4d246a6e428b">










# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
