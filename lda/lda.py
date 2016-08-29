
import numpy as NP
import numpy.random as RNG
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys
import re
import string
from getopt import getopt
from collections import Counter

punctplus = '[' + string.punctuation + ']+' # regex for punct removal
stemmer = PorterStemmer()

##########
# config #
##########
n_iter = 2000   # number of iterations
filelist = ""   # the file containing list of documents
stem = False    # stem the words
eta = 0.01      # smoothing hyperparameter #1
alpha = 0.1     # smoothing hyperparameter #2

# list of option - global variable mappings
getopt_actions = {
        "-N"    :   "n_iter",
        "-T"    :   "filelist",
        "-s"    :   "stem",
        "-e"    :   "eta",
        "-a"    :   "alpha",
}

##################
# option parsing #
##################
opts, args = getopt(sys.argv[1:], "N:T:se:a:")
# If "-N 50" is given as option, the following magically makes n_iter 50.
for opt in opts:
    for act, var in getopt_actions.items():
        if opt[0] == act:
            if type(locals()[var]) is bool:
                locals()[var] = True
            else:
                locals()[var] = type(locals()[var])(opt[1])

T = int(args[0])    # number of topics
V = set()           # the vocabulary
wordcounts = []     # word count for each document
flist = []

if len(filelist) > 0:
    f = open(filelist, "r")
    flist = list(map(str.strip, f.readlines()))
else:
    flist = args[1:]
RNG.shuffle(flist)

# read each document, build the vocabulary and document-term matrix.
for fname in flist:
    f = open(fname, "r")
    s = f.read()
    l = nltk.wordpunct_tokenize(s)
    l = [x for x in l if not re.fullmatch(punctplus, x)]    # punct removal
    l = list(map(str.casefold, l))  # case removal
    l = [x for x in l if x not in stopwords.words('english')]   # stop words
    if stem: l = list(map(stemmer.stem, l)) # stemming
    V |= set(l)
    wordcounts.append(Counter(l))
V = list(V)         # we want the vocab to be ordered now

D_W = []            # document-term matrix
for wordcount in wordcounts:
    w = NP.zeros((len(V),))
    for word, index in zip(V, range(len(V))):
        w[index] = wordcount[word]
    D_W.append(w)
D_W = NP.asarray(D_W).astype(NP.intc)
n_words = D_W.sum() # total number of words

DS = NP.zeros((n_words,), dtype=NP.intc)    # which doc the i-th word is in
WS = NP.zeros((n_words,), dtype=NP.intc)    # which vocab item the i-th word is
idx = 0
for ij in NP.array(NP.nonzero(D_W)).T:      # ij points to non-0 element in D_W
    cnt = D_W[tuple(ij)]
    DS[idx:idx+cnt] = ij[0]
    WS[idx:idx+cnt] = ij[1]
    idx += cnt

ZS = NP.zeros((n_words,), dtype=NP.intc)    # which topic the i-th word is
DZ = NP.zeros((D_W.shape[0], T), dtype=NP.intc) # doc-topic matrix
ZW = NP.zeros((T, D_W.shape[1]), dtype=NP.intc) # topic-vocab matrix
nZ = NP.zeros((T,), dtype=NP.intc)  # number of topic occurrences overall
for i in range(0, n_words):
    ZS[i] = RNG.randint(T)
    DZ[DS[i], ZS[i]] += 1
    ZW[ZS[i], WS[i]] += 1
    nZ[ZS[i]] += 1

# Collapsed Gibbs Sampling goes on here
dist = NP.zeros((T,))   # topic sampling distribution
for _ in range(0, n_iter):
    # note that the sampling is async
    order = NP.arange(0, n_words)
    RNG.shuffle(order)
    for i in order:
        # consider reallocating i-th word so remove its existence
        DZ[DS[i], ZS[i]] -= 1
        ZW[ZS[i], WS[i]] -= 1
        nZ[ZS[i]] -= 1

        # compute topic distribution
        for k in range(0, T):
            dist[k] = (ZW[k, WS[i]] + eta) / (nZ[k] + eta * len(V)) * (DZ[DS[i], k] + alpha)
        dist /= dist.sum()

        # get a new topic for i-th word
        ZS[i] = RNG.choice(NP.arange(0, T), p=dist)
        DZ[DS[i], ZS[i]] += 1
        ZW[ZS[i], WS[i]] += 1
        nZ[ZS[i]] += 1

# DZ now holds the proportion of topics
for i in range(0, D_W.shape[0]):
    sum_k = DZ[i].sum()
    ranks = []
    for k in range(0, T):
        ranks.append((flist[i], k, DZ[i, k], sum_k))
    for rank in sorted(ranks, key=lambda x: x[2], reverse=True):
        if rank[2] != 0:
            print('Doc %s topic %d weight %d/%d' % rank)

# ZW now holds the proportion of words for each topic
for k in range(0, ZW.shape[0]):
    sum_w = ZW[k].sum()
    ranks = []
    for i in range(0, D_W.shape[1]):
        ranks.append((k, V[i], ZW[k, i], sum_w))
    for rank in sorted(ranks, key=lambda x: x[2], reverse=True):
        if rank[2] != 0:
            print('Topic %d word %s weight %d/%d' % rank)

