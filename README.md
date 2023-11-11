# Ex-07-Feature-Selection
# AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
- STEP 1
Read the given Data
- STEP 2
Clean the Data Set using Data Cleaning Process
- STEP 3
Apply Feature selection techniques to all the features of the data set
- STEP 4
Save the data to the file

# PROGRAM

- <B>DATA PREPROCESSING BEFORE FEATURE SELECTION:</B>
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
df.isnull().sum()
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
from sklearn.preprocessing import OrdinalEncoder
gender = ['male','female']
en= OrdinalEncoder(categories = [gender])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"]
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/30c4abf7-02ab-456d-9fbe-4aaa6303d1b4">

- <B>CHECKING NULL VALUES:</B>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/958ca867-e8bc-4f0c-9c30-6e9b8f0ae92d">


- <B>DROPPING UNWANTED DATAS:</B>
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/772589ff-b08f-47f4-bf51-5a1a4776b39d">

- <B>DATA CLEANING:</B>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/d9368867-c4ec-4da0-b947-e9ca1d57a076">

- <B>REMOVING OUTLIERS:</B>
  - Before
<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/2800a653-f79a-477a-81ed-e1c0096e0fb4" height=250 width=300>
 
- 
  - After

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/f7fc0e45-822a-4529-94e1-a9f153e97071" height=250 width=300>


- <B>_FEATURE SELECTION:_</B>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/5ac83c7e-75c7-436d-9fe8-35049a1dfe8e">

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/48a333b2-188a-4721-9474-936e846b9d77">


<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/328828f8-440b-4466-a227-b6d47614e9da">



<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/45a00872-fa32-47da-909f-21d4bf633edf">



- <B>_FILTER METHOD:_</B>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/85f54298-d3e4-4c2a-b7a4-dd864908379a" height=250 width=300>

- <B>_HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:_</B>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/f725dd10-f2e8-4e11-9f2e-b6ab73ac6135">

- <B>_BACKWARD ELIMINATION:_</B>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/951dc4a9-a94e-49ae-a7cc-0d28fb68e937">


- <B>_OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:_</B>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/5eece329-e9b4-44f0-9411-29d992453afc">

- <B>_FINAL SET OF FEATURE:_</B>

<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/7ec9be2c-f22d-4adc-a959-be58866826d7">

- <B>_EMBEDDED METHOD:_</B>


<img src="https://github.com/Janarthanan2/ODD2023-Datascience-Ex-07/assets/119393515/728772fb-2d29-4eb2-bcff-a383dbd125f4" height=250 width=300>



# Inference:
The primary step in feature selection is to remove any unnecessary data. We can do this by using data cleaning and outlier detection to get rid of duplicates and outliers before adding any data that still needs to be cleaned.Preprocessing, cleaning, outlier removal, and a variety of feature selection techniques, including filter techniques (correlation), backward elimination, recursive feature elimination (RFE), and embedded techniques (Lasso), are covered in this sequence of steps.


# RESULT:
Thus, the various methods of feature selection techniques have been performed on a given dataset successfully.
