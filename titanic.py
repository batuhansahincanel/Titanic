# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 18:12:42 2022

@author: Batuhan
"""

#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier



#Importing both train and test sets
trainset = pd.read_csv("train.csv")
testset = pd.read_csv("test.csv")


#Seperating "Survived" column as the y value and dropping it from the train set
y = trainset.iloc[:,1:2].values      #survived or not
trainset = trainset.drop('Survived', axis = 1)

#Assembling train and test sets to make some data preproccesing
dataset = pd.concat([trainset, testset], axis = 0, ignore_index=True)

#Checking For Missing Values and Defining Variables
pClassNan = dataset['Pclass'].isna().sum()
pClass = dataset.iloc[:,1:2].values #ticket class

sexNan = dataset['Sex'].isna().sum()
sex = dataset.iloc[:,3:4].values

sibSpNan = dataset['SibSp'].isna().sum()
sibSp = dataset.iloc[:,5:6].values  # number of Siblings or Supouses 

ageNan = dataset['Age'].isna().sum()
age = dataset['Age'].fillna(dataset['Age'].mean()).values

nameNan = dataset['Name'].isna().sum()
name = dataset.iloc[:,2:3].values

parChNan = dataset['Parch'].isna().sum()
parCh = dataset.iloc[:,6:7].values  # number of Parents or Children

embarkedNan = dataset['Embarked'].isna().sum()
embarked = dataset['Embarked'].fillna('U').values # where the passengers embarked with the ship C=Cherbourg, Q=Queenstown, S=Southampton, U=Unknown

#there were 177 missing age values and all are filled with the mean of the Age column
#there were 2 missing embarking values and both filled with the value of U which means Unknown


#Encoding

ohe = OneHotEncoder()
le = LabelEncoder()

sex = le.fit_transform(sex)

embarked = embarked.reshape(-1,1)
embarked = ohe.fit_transform(embarked).toarray()

df_emb = pd.DataFrame(data=embarked, index= range(1309), columns=['C', 'Q', 'S', 'U'])
print(df_emb)
df_pClass = pd.DataFrame(data=pClass, index= range(1309), columns=['pClass'])
print(df_pClass)
df_sex = pd.DataFrame(data=sex, index= range(1309), columns=['sex(1=M, 0=F)'])
print(df_sex)
df_sibSp = pd.DataFrame(data=sibSp, index= range(1309), columns=['sibSp'])
print(df_sibSp)
df_age = pd.DataFrame(data=age, index= range(1309), columns=['age'])
print(df_age)
df_parCh = pd.DataFrame(data=parCh, index= range(1309), columns=['parCh'])
print(df_parCh)


df = pd.concat([df_age, df_sex, df_pClass, df_sibSp, df_parCh, df_emb], axis=1)
X = df.iloc[0:891,:].values
test = df.iloc[891:,:].values
id_test = dataset.iloc[891:,0]



#Building a Random Forest Classifier Model
classifier_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
classifier_rf.fit(X, y)
#Prediction
y_pred = classifier_rf.predict(test)


#Building a SVM Classifier Model 
from sklearn.svm import SVC
classifier_svc = SVC(probability=True, kernel="rbf")
classifier_svc.fit(X,y)
#Prediction
y_pred2 = classifier_svc.predict(test)


#Building a Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(X, y)
#Prediction
y_pred3 = classifier_lr.predict(test)


#Building a KNN Model
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 10)
classifier_knn.fit(X, y)
#Prediction
y_pred4 = classifier_knn.predict(test)


#Building a NaiveBayes Model
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X, y)
#Prediction
y_pred5 = classifier_nb.predict(test)

#Outputting
output = pd.DataFrame({'PassengerId' : id_test, 'Survived' : y_pred })
output.to_csv('submission.csv', index=False)

output2 = pd.DataFrame({'PassengerId' : id_test, 'Survived' : y_pred2 })
output2.to_csv('submission2.csv', index=False)

output3 = pd.DataFrame({'PassengerId' : id_test, 'Survived' : y_pred3 })
output3.to_csv('submission3.csv', index=False)

output4 = pd.DataFrame({'PassengerId' : id_test, 'Survived' : y_pred4 })
output4.to_csv('submission4.csv', index=False)

output5 = pd.DataFrame({'PassengerId' : id_test, 'Survived' : y_pred5 })
output5.to_csv('submission5.csv', index=False)





