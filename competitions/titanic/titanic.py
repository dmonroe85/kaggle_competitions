""" Kaggle Titanic Survivor Competition
    Tutorial With Numpy
"""
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def read_data(filename):
    return pd.read_csv(filename, header=0)

def clean(data_frames):
    enums = data_frames.copy(deep = True)

    # Deriving and Enumerating complete features
    enums['Sex'], trash = pd.factorize(enums['Sex'])

    enums['Family'] = enums['SibSp'] + enums['Parch']
    enums['NotAlone'] = (enums['Family'] > 0).astype(int)

    enums['Firstname'] = enums['Name'].str.extract('(Mr\. |Miss\. |Master\. |Mrs\. |Don\. |Rev\. |Dr\. |Mme\. |Ms\. |Mlle\. |Major\. |Lady\. |Sir\. |Col\. |Capt\. |Jonkheer\. |the Countess\.[A-Za-z ]*\()([A-Za-z]*)')[1]
    enums['Firstname'], trash = pd.factorize(enums['Firstname'])

    enums['Title'] = enums['Name'].str.extract('(Mr\.|Miss\.|Master\.|Mrs\.|Don\.|Rev\.|Dr\.|Mme\.|Ms\.|Mlle\.|Major\.|Lady\.|Sir\.|Col\.|Capt\.|Jonkheer\.|the Countess\.)')
    enums['Title'], trash = pd.factorize(enums['Title'])

    # Filling in Missing Data
    enums['Embnan'] = 0
    enums.loc[enums['Embarked'].isnull(), 'Embnan'] = 1

    missing = enums.copy(deep=True)
    missing = missing.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Age'], axis=1)

    enums['Agenan'] = 0
    enums.loc[enums['Age'].isnull(), 'Agenan'] = 1

    enums['Cabnan'] = 0
    enums.loc[enums['Cabin'].isnull(), 'Cabnan'] = 1

    enums['Embarked'], trash = pd.factorize(enums['Embarked'])

    enums.loc[enums['Fare'].isnull(), 'Fare'] = 0

    # Drop the remaining columns
    enums = enums.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Age'], axis=1)

    return enums.values

train_frames = read_data('data/train.csv')
test_frames = read_data('data/test.csv')

train_set = clean(train_frames)
test_set = clean(test_frames)

forest = RandomForestClassifier(n_estimators = 1000, n_jobs = -1)
forest = forest.fit(train_set[0::, 1::], train_set[0::, 0])
output = forest.predict(test_set)

fileout = pd.DataFrame()

fileout['PassengerId'] = test_frames['PassengerId']
fileout['Survived'] = output.astype(int)
fileout.to_csv(path_or_buf = 'submissions/first.csv', index = False)