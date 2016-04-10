""" Stopping Point

    Ideas:
    -Investigate address/lat-long differences; why are they different, and how different are they?
    -They give a lot of extra information in the training set - is there a way to use it?
        1.) Divide up the training set into test/train and learn models for features from description, resolution
        2.) use those predictions as extra inputs to the final model
    -Is there an intelligent way to use lat/longs?  Binning into voronoi cells, maybe?
"""

import sys
sys.path.append('../../../Libraries/python_utils/')

from ml_utils.data.handling import csv_to_row_dicts, get_targets, remove_ignored, \
    split_multiclass
from ml_utils.data.sklearn_compatible import PercentileCategorizer
from ml_utils.learners.utils import multiclass_prediction
from ml_utils.metrics.metrics import logloss
from ml_utils.analysis.analysis import val_frequency_hist, analyze_date_format, write_val_hist
from py_utils.utils import is_num

import string, csv, re
from calendar import monthrange

import numpy as np
from sklearn import pipeline, feature_extraction, ensemble, linear_model, tree
from sklearn.cross_validation import LabelShuffleSplit as Splitter

training_file = 'data/train.csv'
test_file = 'data/test.csv'
training,   headers = csv_to_row_dicts(training_file, display=True, row_limit=0)
# test,       _       = csv_to_row_dicts(test_file,     display=True)

training_set = training.values()
# test_set = test.values()

if 0:
    full_set = training_set + test_set

    Addresses = {};    Mismatches = [];    N_mismatches = 0
    for entry in training_set:
        add_data = {'address': '', 'x': '', 'y': ''}
        for key in entry:
            if key in ['address', 'x', 'y']:
                if is_num(entry[key]):
                    add_data[key] = "%.2f" % float(entry[key])
                else:
                    add_data[key] = "%s"   % entry[key]

        if add_data['address'] not in Addresses:
            Addresses[add_data['address']] = [add_data]
        elif add_data not in Addresses[add_data['address']]:
            Addresses[add_data['address']].append(add_data)
        else:
            N_mismatches += 1
            if add_data['address'] not in Mismatches:
                Mismatches = [add_data['address']]
            else:
                Mismatches.append(add_data['address'])

    print N_mismatches
    print len(Addresses)
    print len(Mismatches)

    for m in Mismatches:
        pass

    print analyze_date_format(full_set, 'dates')

    write_val_hist(val_frequency_hist(training_set), print_fields=['category', 'pddistrict', 'resolution', 'dayofweek', 'descript'], filename='results/summary_train.csv')
    write_val_hist(val_frequency_hist(test_set), print_fields=['category', 'pddistrict', 'resolution', 'dayofweek', 'descript'], filename='results/summary_test.csv')
    write_val_hist(val_frequency_hist(full_set), print_fields=['category', 'pddistrict', 'resolution', 'dayofweek', 'descript'], filename='results/summary_full.csv')
    
def preprocess_sf_crime(data):
    table = string.maketrans("", "")
    for entry in data:
        if 'dates' in entry:
            date, time = entry['dates'].split()
            year, month, day = date.split('-')

            if len(year) != 4 or int(month) > 12 or \
              int(day) > monthrange(int(year), int(month)):
                errmsg = "Invalid date format: %s-%s-%s" % (year, month, day)
                raise Exception(errmsg)

            hour, minute, second = time.split(':')

            if int(hour) > 12 or int(minute) > 60 or int(second) > 60:
                errmsg = "Invalid time format: %s:%s:%s" % (hour, minute, second)

            entry['year'] = year
            entry['month'] = month

            entry['hour'] = hour
            del entry['dates']

        if 'x' in entry and is_num(entry['x']):
            entry['x'] = float(entry['x'])

        if 'y' in entry and is_num(entry['y']):
            entry['y'] = float(entry['y'])


target_field = 'category'
ignore_fields = ['descript', 'resolution', 'id',
    # 'dayofweek',
    # 'pddistrict',
    # 'dates',
    'x',
    'y',
    'address',
]

remove_ignored(training_set, ignore_fields)
# remove_ignored(test_set, ignore_fields)

preprocess_sf_crime(training_set)

train_wo_tgts, train_targets_wo_multiclass = get_targets(training_set, target_field)
train_targets = train_targets_wo_multiclass
# train_targets = split_multiclass(train_targets_wo_multiclass)

# Going away from pipeline - not enough need to chain things together.
print "DictVectorizer"
DV = feature_extraction.DictVectorizer(sparse=False)
train_inputs = DV.fit_transform(train_wo_tgts)
print train_inputs.shape

SSS = Splitter(train_targets_wo_multiclass, n_iter=1, test_size=0.5)
for tr, cv in SSS:
    print "Split"
    train_idx, cv_idx = tr, cv
    print train_idx.shape
    print cv_idx.shape

    print "Train Classifier"
    CF = ensemble.RandomForestClassifier(n_jobs=8, verbose=1, n_estimators=66)

    CF.fit(train_inputs[train_idx, :], train_targets[train_idx])

    print "Predictions on training set"
    print logloss(
        np.array(CF.predict_proba(train_inputs[train_idx, :])).T,
        train_targets[train_idx]
    )

    print "Predictions on cross validation set"
    print logloss(
        np.array(CF.predict_proba(train_inputs[cv_idx, :])).T,
        train_targets[cv_idx]
    )

    print "Predictions on the full set"
    print logloss(
        np.array(CF.predict_proba(train_inputs)).T,
        train_targets
    )