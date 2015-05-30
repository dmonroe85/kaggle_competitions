import csv
import re
import numpy as np
import pandas as pd
import pylab as P
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

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

def quantiles(df_in, field,  size=5):
    step = 1.0/(size)
    quant = 0
    for idx in xrange(size):
        quant += step
        lower_lim = df_in[field].dropna().quantile(quant-step)
        upper_lim = df_in[field].dropna().quantile(quant)
        df_in[field + 'bin' + str(idx)] = \
                                       ((df_in[field] > lower_lim) & \
                                        (df_in[field] <= upper_lim)).astype(int)
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

    # Age Analysis
    enums['AgeIsNull'] = pd.isnull(enums['Age']).astype(int)
    median_ages = np.zeros((2,3))
    for i in range(2):
        for j in range(3):
            median_ages[i, j] = enums[ (enums['Sex'] == i) & \
                                       (enums['Pclass'] == (j))]['Age'].dropna().median()
            enums.loc[ (enums['Age'].isnull()) &\
                       (enums['Sex'] == i) & \
                       (enums['Pclass'] == j), 'Age'] = median_ages[i, j]

    enums = quantiles(enums, 'Age', 10)

    # Fare Analysis
    enums.loc[enums['Fare'].isnull(), 'Fare'] = enums['Fare'].dropna().median()
    enums['RodeForFree'] = (enums['Fare'] == 0).astype(int)

    enums = quantiles(enums, 'Fare', 10)

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

    # Misc
    enums['AgeIsEst']  = ((enums['Age'] - enums['Age'].astype(int).astype(float)) == 0.5).astype(int)
    enums['Age*Class'] = enums['Age']*enums['Pclass']
    enums = quantiles(enums, 'Age*Class', 10)
    enums['Sex*Class'] = enums['Sex']*enums['Pclass']
    enums['Fem&1stClass'] = ((enums['Sex'] == 0) & (enums['Pclass'] == 1)).astype(int)
    enums['FamilySize'] = enums['SibSp'] + enums['Parch']
    enums = enum_df(enums, 'FamilySize')
    enums['TrvelAlone'] = (enums['FamilySize'] == 0).astype(int)
    enums['Sex*Alone'] = enums['Sex']*enums['TrvelAlone']
    enums['Class*Fare'] = enums['Pclass']*enums['Fare']
    enums = quantiles(enums, 'Class*Fare', 10)

    # enums.to_csv('cabin.csv')

    passids = enums['PassengerId'].values
    enums = enums.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return enums, passids
    

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_df, garbage = process_csv('data/train.csv')

low_corr_names = train_df.corr().loc[train_df.corr()['Survived'].abs() < 0.05, 'Survived'].index
for idxname in low_corr_names:
    train_df = train_df.drop([idxname], axis=1)

train_data = train_df.values

test_df, ids = process_csv('data/test.csv')
for idxname in low_corr_names:
    test_df = test_df.drop([idxname], axis=1)

test_data = test_df.values


print 'Training...'
clf = Pipeline([
    ('feature_selection', ExtraTreesClassifier()),
    ('classification', RandomForestClassifier())
])
clf = clf.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = clf.predict(test_data).astype(int)

print 'Printing...'
predictions_file = open("eleventh.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

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