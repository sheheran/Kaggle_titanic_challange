#Import libraries
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import csv
import math
import numpy as np
from sklearn import preprocessing, svm
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#importing calssifire modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Load CSV file with pandas
#import training set
trainSet = 'train.csv'
trainNames = ['PassengerID','Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSq', 'Parch', 'Ticket', 'Fare','Cabin', 'Embarked']
trainData = pd.read_csv(trainSet, names=trainNames)
print(trainData.shape)
#import test set
testSet = 'test.csv'
testNames =  ['PassengerID', 'Pclass', 'Name', 'Sex', 'Age', 'SibSq', 'Parch', 'Ticket', 'Fare','Cabin', 'Embarked']
testData = pd.read_csv(testSet, names=testNames)
#print(testData.shape)
print(trainData.head())
#trainData.info()
#testData.info()
#Count of null spaces
#print(trainData.isnull().sum())
#print(testData.isnull().sum())
#bar chart for defined frature
def bar_chart(feature):
	survived = trainData[trainData['Survived']==1][feature].value_counts()
	dead = trainData[trainData['Survived']==0][feature].value_counts()
	df = pd.DataFrame([survived,dead])
	df.index = ['Survived','Dead']
	df.plot(kind='bar',stacked=True, figsize=(10,5))
	return;
bar_chart('Sex')
#plt.show()
bar_chart('Pclass')
#plt.show()
bar_chart('SibSq')
#plt.show()
#Feature engineering
train_test_data = [trainData, testData] #combining train and test data
for dataset in train_test_data:
	dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)

#[a-zA-Z]+ a word consisting of only lathin characters with a length at least one
#print(len(trainData['Title']))
print(trainData['Title'].value_counts())
title_mapping = {"Mr":0, "Miss": 1, "Mrs": 2,
				"Master":3, "Dr":3, "Rev":3, "Col":3, "Major":3, "Mlle":3, "Countess":3 ,
				"Ms":3, "Lady":3, "Jonkheer":3, "Don":3, "Dona": 3, "Mme":3, "Capt": 3, "Sir":3	}
for dataset in train_test_data:
	dataset['Title'] = dataset['Title'].map(title_mapping)
#print(testData.head(5))
#bar_chart('Title')
#plot.show()
#Drop the name class
trainData.drop('Name', axis=1, inplace=True)
testData.drop('Name', axis=1, inplace=True)
#print(trainData.head(5))
#mapping gender
geneder_mapping = {"male": 0, "female":1}
for dataset in train_test_data:
	dataset['Sex'] = dataset['Sex'].map(geneder_mapping)
# Fill missing age with median age for each title (Mr, Mrs, Miss, Others)
trainData["Age"].fillna(trainData.groupby("Title")["Age"].transform("median"),inplace=True)
testData["Age"].fillna(testData.groupby("Title")["Age"].transform("median"),inplace=True)
#plot classification of age
#child:0, young:1, adult:2, mid-age:3, senior:4
for dataset in train_test_data:
	dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
	dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
	dataset.loc[ (dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
	dataset.loc[ (dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
	dataset.loc[ (dataset['Age'] > 62), 'Age'] = 4
#print(trainData.head(5))
#Embarked
Pclass1 = trainData[trainData['Pclass']==1]['Embarked'].value_counts()
Pclass2 = trainData[trainData['Pclass']==2]['Embarked'].value_counts()
Pclass3 = trainData[trainData['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2ed class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#fill S for the missing value most pasengrs from all class are from S
for dataset in train_test_data:
	dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = { "S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
	dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
# fill missing fare with median fare for each Pclass
trainData["Fare"].fillna(trainData.groupby("Pclass")["Fare"].transform("median"), inplace=True)
testData["Fare"].fillna(testData.groupby("Pclass")["Fare"].transform("median"),inplace=True)
for dataset in train_test_data:
	dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
	dataset.loc[ (dataset['Fare'] >17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
	dataset.loc[ (dataset['Fare'] >30) & (dataset['Fare'] <=100), 'Fare'] = 2,
	dataset.loc[ (dataset['Fare']> 100), 'Fare'] = 3
# cabin, count of cabins
trainData.Cabin.value_counts()
#extracting the first character
for dataset in train_test_data:
	dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = trainData[trainData['Pclass']==1]['Cabin'].value_counts()
Pclass2 = trainData[trainData['Pclass']==2]['Cabin'].value_counts()
Pclass3 = trainData[trainData['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2ed class','3ed class']
df.plot(kind='bar' ,stacked=True, figsize=(10,5))
#feature scaling
cabin_mapping = {"A" : 0, "B" : 0.4, "D" : 0.8, "E" : 1.2, "F" : 1.6, "F" : 2, "G" : 2.4, "T" : 2.8}
for dataset in train_test_data:
	dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
#fill missing fare with median for each Pclass
trainData['Cabin'].fillna(trainData.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
testData['Cabin'].fillna(testData.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
#Familisize, combining familisize with parents and siblings
trainData["Familisize"] = trainData["SibSq"] + trainData["Parch"] + 1
testData["Familisize"] = testData["SibSq"] + testData["Parch"] + 1
#chart
#facet = sns.FacetGrid(trainData, hue="Survived", aspect=4)
#facet.map(sns.kdeplot,'Familisize',shade=True)
#facet.set(xlim=(0, train['Familisize'].max()))
#facet.add_legend()
#plt.xlim(0)
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6 : 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
	dataset['Familisize'] = dataset['Familisize'].map(family_mapping)
features_drop = ['Ticket', 'SibSq', 'Parch']
trainData = trainData.drop(features_drop, axis=1)
testData = testData.drop(features_drop, axis=1)
trainData = trainData.drop(['PassengerID'], axis=1)
train_data = trainData.drop('Survived', axis=1)
target = trainData['Survived']
#print(trainData.info())
#print(train_data.head(10))
#print(testData.head(10))
#testing with just train set
X_train, X_test, Y_train, Y_test = train_test_split(train_data,target, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
#fit a model
#clf = LinearRegression()
#model = clf.fit(X_train,Y_train)
#accuracy = clf.score(X_test,Y_test)
#print(accuracy)

print(train_data.info())

#Cross Validation (K-fold)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

#knn
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
clf.fit(X_train, Y_train)
'''
#example
example_measures = np.array([])
prediction =clf.predict(example_measures)
print(prediction)
'''