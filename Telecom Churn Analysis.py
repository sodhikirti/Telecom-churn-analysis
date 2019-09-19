# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:21:51 2019

@author: Kirti Sodhi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

df=pd.read_excel(r'C:\Users\Kirti Sodhi\Desktop\telco.xlsx')

##### Explorartory Data Analysis
df.describe(include='all')
df.columns.to_series().groupby(df.dtypes).groups
#df.hist()
df.isnull().sum(axis=0)

############Handling Missing Values
mean_LTF=df['Log-toll free'].mean()
df['Log-toll free']=df['Log-toll free'].replace(np.nan,mean_LTF)
mean_LE=df['Log-equipment'].mean()
df['Log-equipment']=df['Log-equipment'].replace(np.nan,mean_LE)
mean_LCC=df['Log-calling card'].mean()
df['Log-calling card']=df['Log-calling card'].replace(np.nan,mean_LCC)
mean_LW=df['Log-wireless'].mean()
df['Log-wireless']=df['Log-wireless'].replace(np.nan,mean_LW)


#########Handling Categorical Data
le=preprocessing.LabelEncoder()
categorical_feature_mask = df.dtypes==object
categorical_cols = df.columns[categorical_feature_mask].tolist()
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

######Droping variables to avoid Multicollinearity
df=df.drop(['Log-long distance','Long distance last month','Equipment rental','Long distance over tenure','Wireless last month','Wireless service'],axis=1)
######### Building a Corelation plot to compute corelations between Churn and Independent Variables
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(40,25)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.show()

#######Splitting the data into test and train 
Y=df.iloc[:,35]
X=df.drop(['Churn within last month'],axis=1)
feat_labels=X
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

###########Building Machine learning pipelines Logistic Regression, SVM, Random Forest
######Logistic Regression
model = LogisticRegression()
result = model.fit(X_train, y_train)
prediction_test = model.predict(X_test)
#####Accuracy of Logistic Regression and weights of the features used in the model
print ('Accuracy of Logistic Regression',metrics.accuracy_score(y_test, prediction_test))
print(classification_report(y_test, prediction_test))
weights = pd.Series(model.coef_[0],index=X.columns.values)
print(weights.sort_values(ascending = False))
#######Confusion matrix for Logistic Regression
confusion_matrix = confusion_matrix(y_test, prediction_test)
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)
#########Random Forest
randomForest = RandomForestClassifier()
randomForest.fit(X_train, y_train)
prediction_test1 = randomForest.predict(X_test)
#####Accuracy of Random Forest
print ('Accuracy of Random Forest',metrics.accuracy_score(y_test, prediction_test1))

#######Checking for most important feauters using Random Forest
for feature in (zip(feat_labels, randomForest.feature_importances_)):
    print(feature)
features = feat_labels
feat_importances = pd.Series(randomForest.feature_importances_, index=features.columns)
feat_importances.nlargest(10).plot(kind='barh')

#######Building Model using the top feauters and Comparing the result with the previous model's accuracies
top_feat=feat_importances.nlargest(20)
top_frame=top_feat.to_frame()
top_X=top_frame.index
feat_new=pd.DataFrame(top_X)
df_new=df[top_X]

X_train, X_test, y_train, y_test = train_test_split(df_new, Y, test_size=0.30, random_state=101)
randomForest = RandomForestClassifier()
randomForest.fit(X_train, y_train)
prediction_test1 = randomForest.predict(X_test)
print ('Accuracy of Random Forest1',metrics.accuracy_score(y_test, prediction_test1))
model = LogisticRegression()
result = model.fit(X_train, y_train)
prediction_test = model.predict(X_test)
print ('Accuracy of Logistic Regression',metrics.accuracy_score(y_test, prediction_test))
