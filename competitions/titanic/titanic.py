import sys
sys.path.append('../../../Libraries/python_utils/')
from py_utils.utils import is_num, checkEqual

from ml_utils.data.handling import csv_to_row_dicts, get_targets
from ml_utils.analysis.analysis import compare_data
from ml_utils.data.sklearn_compatible import \
    PercentileCategorizer, LowCountTrimmer, CollaborativeFilter
from ml_utils.metrics.metrics import pearson_matrix

import csv, string, copy
from sklearn import preprocessing, feature_extraction, feature_selection, \
                    ensemble, tree, linear_model, svm, decomposition, manifold, neural_network, neighbors, \
                    pipeline, grid_search
from functools import partial
from collections import Counter
import numpy as np
import scipy as sp

# Preprocessing
def preprocessing_titanic(data, ignored=[], target_field=''):
    table = string.maketrans("", "")
    # First pass, preprocess input data
    for entry in data:
        # Ignoring Features
        for field in ignored:
            if field in entry and field != target_field:
                del entry[field]

        # Deriving New Features
        if 'age' not in ignored:
            entry['age_est'] = ''
            if 'age' in entry and entry['age']:
                entry['age_est'] = float(entry['age'])
                if entry['age'].endswith('.5'):
                    entry['age'] = ''
                else:
                    entry['age'] = float(entry['age'])
            else:
                entry['age'] = ''

        if 'sibsp' not in ignored and 'parch' not in ignored:
            entry['family'] = ''
            if 'sibsp' in entry and 'parch' in entry and entry['sibsp'] and entry['parch']:
                entry['family'] = "%s" % (int(entry['sibsp']) + int(entry['parch']))

        if 'cabin' not in ignored:
            entry['cabin_deck'] = ''
            entry['cabin_prefix'] = ''
            entry['cabin_port_starboard'] = ''
            if 'cabin' in entry:
                if entry['cabin']:
                    cabins = entry['cabin'].split()

                    entry['cabin_deck'] = Counter(map(lambda x: x[0], cabins)).most_common(1)[0][0]

                    single_letter = filter(lambda x: len(x) == 1, cabins)
                    entry['cabin_prefix'] = single_letter[0] if single_letter else ''

                    rooms = filter(lambda x: len(x) > 1 and is_num(x[1:]), cabins)
                    port_starboard = Counter(
                        map(lambda x: "%s" % (float(x[1:]) % 2), rooms)
                    ).most_common(1)[0][0] if rooms else ''

                    entry['cabin_port_starboard'] = port_starboard

                del entry['cabin']
                

        if 'ticket' not in ignored:
            entry['ticket_number'] = ''
            entry['ticket_text'] = ''
            if 'ticket' in entry:
                if entry['ticket']:
                    split = entry['ticket'].split()

                    if len(split) > 1:
                        number = split[-1]
                        text = ''.join(split[:-1]).translate(table, string.punctuation).lower()
                    elif len(split) == 1:
                        number = split[0]
                        text = ''
                    else:
                        number = ''
                        text = ''

                    entry['ticket_number'] = number
                    entry['ticket_text'] = text

                del entry['ticket']

        if 'name' not in ignored:
            if 'name' in entry:
                if entry['name']:
                    name_tokens = entry['name'].translate(table, string.punctuation).lower().split()
                    for t in name_tokens:
                        entry['name=' + t] = 1.0

                del entry['name']

target_field = 'survived'

training_file = 'data/train.csv'
test_file = 'data/test.csv'
other_file = 'data/titanic3.csv'

training,   headers = csv_to_row_dicts(training_file,   ['name', 'ticket'])
test,       _       = csv_to_row_dicts(test_file,       ['name', 'ticket'])
other,      _       = csv_to_row_dicts(other_file,      ['name', 'ticket'])

compare_data(other, training)
cheat_test = compare_data(other, test)

analysis_set = (training.values())
cheat_set = (cheat_test.values())
full_set = analysis_set + cheat_set

# Delete Uninteresting Variables
ignore_fields = ['boat', 'home.dest', 'body', 'passengerid', 'survived',
    # 'sex',
    # 'pclass',
    # 'parch',
    # 'sibsp',
    # 'embarked',
    # 'fare',
    # 'age',
    # 'cabin',
    # 'ticket',
    # 'name',
]

# Preprocess Data
preprocessing_titanic(analysis_set, ignore_fields, target_field)
preprocessing_titanic(cheat_set, ignore_fields, target_field)

# Get Targets
balanced_set, targets = get_targets(analysis_set, target_field, balance=False)
_,      cheat_targets = get_targets(cheat_set, target_field, balance=False)

print "\nTarget mean = %s" % np.average(targets)
print "Cheat Target mean = %s\n" % np.average(cheat_targets)

DV = feature_extraction.DictVectorizer(sparse=False)

# CF = CollaborativeFilter()
# collaborated = CF.fit_transform(balanced_set, targets)

PL = pipeline.Pipeline(steps=[
    ("collab",          CollaborativeFilter(L=1)),
    ("pctcats",         PercentileCategorizer({'fare': 10, 'age': 10, 'ticket_number': 10})),
    # ("lowcount",        LowCountTrimmer(threshold=0, criteria='field')),

    ("dict2vec",        DV),
    # ("PCA",             decomposition.PCA())
    # ("selectk",         feature_selection.SelectKBest(k=10)),

    # ("logistic",        linear_model.LogisticRegression(penalty='l1', C=1.0, solver='liblinear', class_weight='balanced')),
    # ("dectree",         tree.DecisionTreeClassifier()),
    ("randforest",      ensemble.RandomForestClassifier(n_estimators=10000, n_jobs=8)),
])

if 1:
    PL.fit(balanced_set, targets)
    print PL.score(balanced_set, targets)
    print PL.score(cheat_set, cheat_targets)


if 0:
    pct_cat_vals = [1, 2, 4, 10]
    pct_cat_settings = [{'fare': x, 'age': y, 'ticket_number': z} for x in pct_cat_vals for y in pct_cat_vals for z in pct_cat_vals]

    GS = grid_search.GridSearchCV(
            PL,
            param_grid={
                # 'pctcats__features': pct_cat_settings,
                # 'lowcount__threshold': [500],                                            # 1
                # 'selectk__k': ['all'],                                                  # all

                # 'logistic__C': [.01, .1, 1, 10, 100],

                # 'randforest__n_estimators': [10, 100, 1000],                            # 50
                # 'randforest__max_features': ['sqrt', 'log2', None, .2, .4, .6, .8],     # log2
                'randforest__class_weight': ['balanced'],   # balanced_subsample

                # 'dectree__max_features': [.5, 'log2', 'sqrt'],
                # 'dectree__max_depth': [None],
                # 'dectree__min_samples_split': [2],
                # 'dectree__min_samples_leaf': [1],
                # 'dectree__presort': [True],
            },
            verbose=1,
            n_jobs=4,
    )

    GS.fit(balanced_set, targets)
    predictions = GS.predict(balanced_set)
    print GS.score(balanced_set, targets)
    print GS.score(cheat_set, cheat_targets)
    print GS.best_params_

if 0:
    cat_inputs = copy.deepcopy(inputs)
    cat_cheat  = copy.deepcopy(cheat_inputs)

    CL = [
        # ("Linear",      linear_model.LinearRegression(fit_intercept=True)),
        # ("Logistic",    linear_model.LogisticRegression(penalty='l2', C=1.0, solver='liblinear')),
        # ("DecTree",     tree.DecisionTreeClassifier()),
        # ("RandForest",  ensemble.RandomForestClassifier(n_estimators=10, class_weight='balanced')),
        # ("KernelSVM",   svm.SVC(gamma=.1, probability=False)),
        # ("MLP_NN",      neural_network.MLPClassifier()),
        # ("KNN",         neighbors.KNeighborsClassifier()),
        # ("GBC",         ensemble.GradientBoostingClassifier()),
        # ("ADABoost",    ensemble.AdaBoostClassifier(n_estimators=100)),
        # ("Bagger",      ensemble.BaggingClassifier(neighbors.KNeighborsClassifier()))
    ]

    for tup in CL:
        print ""
        print inputs.shape
        print cheat_inputs.shape

        text, classifier = tup
        classifier.fit(cat_inputs, targets)
        print ""
        print text
        print "Train: %s" % classifier.score(cat_inputs, targets)
        print "Test : %s" % classifier.score(cat_cheat, cheat_targets)

        predictions = classifier.predict(cat_inputs)
        cheat_pred  = classifier.predict(cat_cheat)

        # cat_inputs  = add_predictions(cat_inputs, predictions)
        # cat_cheat   = add_predictions(cat_cheat, cheat_pred)

if 0:
    pearson_vals = pearson_matrix(cat_inputs, targets)
    chi2_vals = feature_selection.chi2(cat_inputs, targets)
    f_vals = feature_selection.f_classif(cat_inputs, targets)

    score_hdrs = ['Feat', 'Corr', 'P', 'IG', 'Chi2', 'P', 'F-Score', 'P']
    scores = [
        DV.feature_names_ ['Lin', 'Log', 'DecTree', 'RF'],
        pearson_vals[0],
        pearson_vals[1],
        IG_matrix(inputs, targets),
        list(chi2_vals[0]),
        list(chi2_vals[1]),
        list(f_vals[0]),
        list(f_vals[1]),
    ]

    W = 35
    print_row(score_hdrs, W)
    for s in sorted(zip(*scores), key=lambda x: abs(x[1]), reverse=True):
        print_row(s, W)

if 0:
    PCA = decomposition.PCA(n_components=3)
    pca_out = PCA.fit_transform(inputs)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_out[targets==1.0, 0], pca_out[targets==1.0, 1], pca_out[targets==1.0, 2], c='b')
    ax.scatter(pca_out[targets==0.0, 0], pca_out[targets==0.0, 1], pca_out[targets==0.0, 2], c='r')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    plt.scatter(pca_out[targets==1.0, 0], pca_out[targets==1.0, 1], c='b')
    plt.scatter(pca_out[targets==0.0, 0], pca_out[targets==0.0, 1], c='r')
    plt.savefig('pca.png')

    print dir(manifold)
    TSNE = manifold.TSNE(n_components=3)
    tsne_out = TSNE.fit_transform(inputs)

    plt.scatter(tsne_out[targets==1.0, 0], tsne_out[targets==1.0, 1], c='b')
    plt.scatter(tsne_out[targets==0.0, 0], tsne_out[targets==0.0, 1], c='r')
    plt.savefig('tsne.png')
