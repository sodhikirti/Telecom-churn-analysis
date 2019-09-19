# Telecom-churn-analysis
Predicting future churn in the telecom industry by analyzing the historic data

Libraries imported
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
  
  Steps Involved:
  
1.	Import the file

2.	Exploratory Data Analysis

3.	Data Pre-processing

a.	Handling missing values

b.	Handling categorical values

c.	Checking Multicollinearlity

4.	Building Machine learning models

a.	Splitting into test and train data sets

b.	Models used: Logistic Regression, Random Forest

5.	Evaluating Models

6.	Selecting only the top 20 important features using Random Forest classifier

7.	Comparing the accuracy of the models with all the features and with only top 20 features used

 
 
  
