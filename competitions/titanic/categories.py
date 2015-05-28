""" Kaggle Titanic Survivor Competition
    Tutorial With Numpy
"""
import csv, re

def surv_count(problist, idx, surv):
    problist[idx][0] += surv
    problist[idx][1] += 1

def port_or_starboard(cabnums):
    """ The cabins on the ship are numbered even for one side and odd for the
    other.
    """
    sidelist = []
    for cab in cabnums:
        # If there is not remainder when divided by 2, the cabin is even
        if not int(cab) % 2:
            sidelist.append(0)
        else:
            sidelist.append(1)

    return int(round(sum(sidelist)/(len(sidelist) + 0.0)))

def print_train_probs(problist, name):
    print "\n" + name
    N = 0
    for entry in problist:
        N += entry[1]

    for idx in xrange(len(problist)):
        prob_survived = problist[idx][0]/(problist[idx][1] + 0.0)
        pct_occurred = problist[idx][1]/(N + 0.0)
        print '%s: %s: %s' % \
            (idx+1, prob_survived, pct_occurred)

train_csv = csv.reader(open('data/train.csv', 'rb'))
test_csv  = csv.reader(open('data/testsurvcol.csv', 'rb'))
header = train_csv.next()

print header

Pclass = [[0, 0], [0, 0], [0, 0]]
Sex = [[0, 0], [0, 0]]
CabListed = [[0, 0], [0, 0]]
NumCabs = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
CabF = [[0, 0], [0, 0], [0, 0]]
Deck = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
Side = [[0, 0], [0, 0], [0, 0]]
Embarked = [[0, 0], [0, 0], [0, 0], [0, 0]]

sexdict = {'male': 0, 'female': 1}
embdict = {'': 0, 'C': 1, 'Q': 2, 'S': 3}
dckdict = {'': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}

for row in train_csv:
    surv = int(row[1])
    pcls = int(row[2])
    sex  = sexdict[row[4]]
    cabstring = row[10]
    if len(cabstring) == 0:
        deckstring = cabstring
        surv_count(CabF, 0, surv)
        surv_count(Side, 0, surv)
        surv_count(CabListed, 0, surv)
        surv_count(NumCabs, 0, surv)
    else:
        if cabstring.startswith('F '):
            surv_count(CabF, 1, surv)
            cabstring = re.sub('F ', '', cabstring)
        else:
            surv_count(CabF, 2, surv)

        deckstring = cabstring[0]

        if re.search('^\w\d', cabstring):
            cabnums = re.sub('[A-Z]', '', cabstring).split()
            surv_count(Side, port_or_starboard(cabnums) + 1, surv)
            surv_count(NumCabs, len(cabnums), surv)
        else:
            surv_count(Side, 0, surv)
            surv_count(NumCabs, 1, surv)

        CabListed[1][0] += surv
        CabListed[1][1] += 1

    dck  = dckdict[deckstring]
    embk = embdict[row[11]]

    surv_count(Pclass, pcls-1, surv)
    surv_count(Sex, sex, surv)
    surv_count(Deck, dck, surv)
    surv_count(Embarked, embk, surv)

print_train_probs(Pclass, 'Pclass')
print_train_probs(Sex, 'Sex')
print_train_probs(CabListed, 'CabListed')
print_train_probs(NumCabs, 'NumCabs')
print_train_probs(CabF, 'CabF')
print_train_probs(Side, 'Side')
print_train_probs(Deck, 'Deck')
print_train_probs(Embarked, 'Embarked')