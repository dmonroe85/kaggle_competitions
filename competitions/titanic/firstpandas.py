import csv
import re
import numpy as np
import pandas as pd
import pylab as P
from sklearn.ensemble import RandomForestClassifier

# Helper Functions
def enum_df(df_in, field, outname='', regex='', extract_idx=0):
    if outname == '':
        outname = field

    map_dict = {}
    enum_idx = 0
    if regex == '':
        unique_items = np.sort(df_in[field].unique())
    else:
        unique_items = np.sort(df_in[field].str.extract(regex)[extract_idx].unique())

    for item in unique_items:
        if pd.isnull(item):
            if regex == '':
                df_in[outname + 'IsNull'] = pd.isnull(df_in[field]).astype(int)
            else:
                df_in[outname + 'IsNull'] = pd.isnull(df_in[field].str.extract(regex)[extract_idx]).astype(int)
        else:
            if regex == '':
                df_in[outname + 'Is' + str(item)] = \
                    (df_in[field] == item).astype(int)
            else:
                df_in[outname + 'Is' + str(item)] = \
                    (df_in[field].str.extract(regex) == \
                     item)[extract_idx].astype(int)

        map_dict[item] = enum_idx
        enum_idx += 1

    if regex == '':
        df_in[outname] = df_in[field].map(map_dict).astype(int)
    else:
        df_in[outname] = df_in[field].str.extract(regex)[extract_idx].map(map_dict)

    return df_in

def cabin_location(df_in, field):
    port_stara = []
    porta = []
    stara = []
    rooma = []
    for val, deck in zip(df_in.loc[df_in[field] == 1, 'Cabin'].values, \
                         df_in.loc[df_in[field] == 1, 'Deck'].values):
        valarray = np.array(re.sub('[A-G]', '', val).split()).astype(int)
        rooma.append(np.average(valarray))
        if deck == 5:
            porta.append(0)
            stara.append(1)
            port_stara.append(2)
        else:
            numport = 0
            numstar = 0
            for num in valarray:
                if num % 2:
                    numstar += 1
                else:
                    numport += 1

            if numstar > numport:
                porta.append(0)
                stara.append(1)
                port_stara.append(2)
            else:
                porta.append(1)
                stara.append(0)
                port_stara.append(1)

    df_in.loc[df_in[field] == 1, 'RoomNumber'] = np.array(rooma)
    df_in.loc[df_in[field] == 1, 'Port'] = np.array(porta)
    df_in.loc[df_in[field] == 1, 'Starboard'] = np.array(stara)
    df_in.loc[df_in[field] == 1, 'PortStar'] = np.array(port_stara)

    return df_in

# start of script
def process_csv(filename):
    df = pd.read_csv(filename, header=0)

    enums = df.copy(deep=True)

    # Enumerating Values and creating new columns
    enums = enum_df(enums, 'Sex')
    enums = enum_df(enums, 'Pclass')
    enums = enum_df(enums, 'Embarked')
    enums = enum_df(enums, 'SibSp')
    enums = enum_df(enums, 'Parch')

    enums['AgeIsNull'] = pd.isnull(enums['Age']).astype(int)
    median_ages = np.zeros((2,3))
    for i in range(2):
        for j in range(3):
            median_ages[i, j] = enums[ (enums['Sex'] == i) & \
                                       (enums['Pclass'] == (j))]['Age'].dropna().median()
            enums.loc[ (enums['Age'].isnull()) &\
                       (enums['Sex'] == i) & \
                       (enums['Pclass'] == j), 'Age'] = median_ages[i, j]

    enums.loc[enums['Fare'].isnull(), 'Fare'] = enums['Fare'].dropna().median()
    enums['RodeForFree'] = (enums['Fare'] == 0).astype(int)
    edp1 = enums.loc[enums['Pclass'] == 1, 'Fare'].describe()
    # print edp1
    # print enums.loc[enums['Pclass'] == 2, 'Fare'].describe()
    # print enums.loc[enums['Pclass'] == 3, 'Fare'].describe()

    # Misc
    enums['AgeIsEst']  = ((enums['Age'] - enums['Age'].astype(int).astype(float)) == 0.5).astype(int)
    enums['Age*Class'] = enums['Age']*enums['Pclass']
    enums['Sex*Class'] = enums['Sex']*enums['Pclass']
    enums['Fem&1stClass'] = ((enums['Sex'] == 0) & (enums['Pclass'] == 1)).astype(int)
    enums['FamilySize'] = enums['SibSp'] + enums['Parch']
    enums['TrvelAlone'] = (enums['FamilySize'] == 0).astype(int)
    enums['Sex*Alone'] = enums['Sex']*enums['TrvelAlone']
    enums['Class*Fare'] = enums['Pclass']*enums['Fare']

    # Cabin Analysis
    enums['CabinNull'] = pd.isnull(enums['Cabin']).astype(int)
    enums = enum_df(enums, 'Cabin', 'Deck', '([A-G])([1-9]+)', 0)
    enums['PrefF'] = (enums['Cabin'].str.extract('([F]) (.*)')[0] != 'F').astype(int)

    enums['Has4Cabins'] = (enums['Cabin'].str.extract('([A-G][1-9]+ [A-G][1-9]+ [A-G][1-9]+ [A-G][1-9]+)').isnull() == False).astype(int)
    enums['Has3Cabins'] = (enums['Cabin'].str.extract('([A-G][1-9]+ [A-G][1-9]+ [A-G][1-9]+)').isnull() == False).astype(int) - enums['Has4Cabins']
    enums['Has2Cabins'] = (enums['Cabin'].str.extract('([A-G][1-9]+ [A-G][1-9]+)').isnull() == False).astype(int) - enums['Has3Cabins']
    enums['Has1Cabin'] = (enums['Cabin'].str.extract('([A-G][1-9]+)').isnull() == False).astype(int) - enums['Has2Cabins']

    enums['AtLeast3Cabins'] = (enums['Cabin'].str.extract('([A-G][1-9]+ [A-G][1-9]+ [A-G][1-9]+)').isnull() == False).astype(int)
    enums['AtLeast2Cabins'] = (enums['Cabin'].str.extract('([A-G][1-9]+ [A-G][1-9]+)').isnull() == False).astype(int)
    enums['AtLeast1Cabin'] = (enums['Cabin'].str.extract('([A-G][1-9]+)').isnull() == False).astype(int)

    enums['RoomNumber'] = 0
    enums['Port'] = 0
    enums['Starboard'] = 0
    enums['PortStar'] = 0

    enums = cabin_location(enums, 'Has4Cabins')
    enums = cabin_location(enums, 'Has3Cabins')
    enums = cabin_location(enums, 'Has2Cabins')
    enums = cabin_location(enums, 'Has1Cabin')

    # enums.to_csv('cabin.csv')


    # enums['Age'].hist()
    # P.show()
    # enums['Pclass'].hist()
    # P.show()
    # enums['Age*Class'].hist()
    # P.show()
    passids = enums['PassengerId'].values
    enums = enums.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return enums, passids
# print enums.corr().loc[enums.corr()['Survived'].abs() > 0.3, 'Survived']

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_df, garbage = process_csv('data/train.csv')
train_data = train_df.values
test_df, ids = process_csv('data/test.csv')
print test_df.isnull().sum()[test_df.isnull().sum() > 0]
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

print 'Printing...'
predictions_file = open("second_submission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

# print enums[ enums['EmbarkedIsNull'] == 1][['Sex', 'Pclass', 'Age', 'Embarked']].head(10)

""" Create Estimator Models to fill in columns with missing parameters 
    -For each missing variable, fill in the one with the least about of missing
     entries first.
    -Once that is filled, use it as an input to the next variable with missing
     entries.
    -Complete until finished.
"""

"""
Number of letters in name
Number of names
Number of Vowels
Number of Consonants
Ratio of Vowels to Consonants"""