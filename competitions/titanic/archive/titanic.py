import sys
sys.path.append('../../../Libraries/python_utils/')

import numpy as np
import math

from ml_utils import IG_matrix, correlation_matrix


class Categories(object):
    def __init__(self, title):
        self.title = title
        self.max_idx = 1 # 0 is reserved for missing data
        self.classdct = {'missing': 0}
        self.classes = []

    def categorize(self, text, new_cat=True):
        stxt = text.strip()
        if new_cat and stxt not in self.classdct:
            self.classdct[stxt], self.max_idx = self.max_idx, self.max_idx + 1

        if stxt in self.classdct:
            self.classes.append(self.classdct[stxt])
        else:
            self.classes.append(self.classdct['missing'])

    def tokens(self):
        token_list = []
        for item in self.classes:
            L = len(self.classdct.keys())
            if L <= 3:
                # if there are 3 or less entries that means we have a binary
                # feature and the only values will be 1 or 2.
                row = [0] * 2
                if item > 0:
                    row[1] = item - 1
                else:
                    row[0] = 1
            else:
                row = [0] * L
                row[item] = 1
            token_list.append(row)
        return token_list


train_file = 'data/train.csv'
test_file = 'data/train.csv'

with open(train_file, 'r') as infile:
    train_lines = infile.readlines()

surv = []; inputs = []
Pcl = Categories('pcl')
Lst = Categories('lst')
Ttl = Categories('ttl')
Nck = Categories('nck')
Sex = Categories('sex')
Age = Categories('age')
Ssp = Categories('ssp')
Pch = Categories('pch')
Emb = Categories('emb')
Fam = Categories('fam')
Aln = Categories('aln')
Fcb = Categories('fcb')
Tcb = Categories('tcb')
Dcb = Categories('dcb')
Dck = Categories('dck')
Prt = Categories('prt')

age_func = lambda x: str(int(float(x)/17))

fare_series = []
famsize_series = []
age_series = []

for idx, line in enumerate(train_lines):
    lsplit = line.split(',')

    if idx == 0:
        train_headers = lsplit
    else:
        # Collecting the target
        surv.append(int(lsplit[1]))

        # Collecting input data
        Pcl.categorize(lsplit[2])
        Lst.categorize(lsplit[3])
        Ttl.categorize(lsplit[4].split('.')[0].strip())
        Nck.categorize(str('("' in lsplit[4]))
        Sex.categorize(lsplit[5])
        age = lsplit[6]
        if age:
            Age.categorize(age_func(age))
        else:
            Age.categorize(' ')
        ssp = lsplit[7]
        Ssp.categorize(ssp)
        pch = lsplit[8]
        Pch.categorize(pch)
        cab = lsplit[11].strip()
        if cab:
            Fcb.categorize(str('F ' in cab))
            cab = cab.replace('F ', '')
            Dcb.categorize(str(cab == 'D'))
            Tcb.categorize(str(cab == 'T'))
            csplit = cab.split()
            if cab != 'D' and cab != 'T' and csplit:
                Dck.categorize(csplit[0][0])
                Prt.categorize(str(int(csplit[0][1:])%2))
            else:
                Dck.categorize(' ')
                Prt.categorize(' ')
        else:
            Fcb.categorize(' ')
            Dcb.categorize(' ')
            Tcb.categorize(' ')
            Dck.categorize(' ')
            Prt.categorize(' ')

        Emb.categorize(lsplit[12])

        # Generating New Inputs
        fam = float(ssp) + float(pch)
        Fam.categorize(str(fam))                                    # Total number of family on board
        Aln.categorize('1' if float(ssp) or float(pch) else '0')    # Travelling alone or not

inputs = [sum(x, []) for x in zip(
                                  Pcl.tokens(),
                                  Lst.tokens(),
                                  Ttl.tokens(),
                                  Nck.tokens(),
                                  Sex.tokens(),
                                  Age.tokens(),
                                  Ssp.tokens(),
                                  Pch.tokens(),
                                  Emb.tokens(),
                                  Fam.tokens(),
                                  Aln.tokens(),
                                  Fcb.tokens(),
                                  Dcb.tokens(),
                                  Tcb.tokens(),
                                  Dck.tokens(),
                                  Prt.tokens()
                                  )]

for idx, tup in enumerate(zip(IG_matrix(inputs, surv), correlation_matrix(inputs, surv))):
    ig, cor = tup
    print idx, ig, cor