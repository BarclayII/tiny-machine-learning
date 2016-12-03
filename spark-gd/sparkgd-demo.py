
from pyspark import SparkContext
import sparkgd
import numpy as NP
import numpy.random as RNG

sc = SparkContext()
sc.addPyFile('sparkgd.py')

ca = RNG.uniform(-1, 1, [2, 2])
cb = RNG.uniform(-1, 1, [2, 2])

neg = RNG.multivariate_normal([-1, -1], ca.dot(ca.T), 100)
pos = RNG.multivariate_normal([1, 1], cb.dot(cb.T), 100)

def generate_data(ca, cb):
    neg = RNG.multivariate_normal([-1, -1], ca.dot(ca.T), 10000)
    pos = RNG.multivariate_normal([1, 1], cb.dot(cb.T), 10000)
    X = NP.concatenate([neg, pos])
    Y = NP.array([-1] * 10000 + [1] * 10000)
    return zip(X, Y)

dataset = (sc.parallelize(range(0, 20)).cache()
           .flatMap(lambda i: generate_data(ca, cb)))
dataset_count = dataset.count()

w = RNG.uniform(-1, 1, 2)
b = RNG.uniform(-1, 1)
model = {'w': w, 'b': b}

def grad(data_point, model):
    X, Y = data_point
    pred = model.value['w'].dot(X) + model.value['b']
    loss = (Y - pred) ** 2
    grad = {'w': (pred - Y) * X, 'b': (pred - Y)}
    return loss, grad

def desc(model, grads):
    model['w'] -= 0.1 * grads['w']
    model['b'] -= 0.1 * grads['b']

for i in range(0, 100):
    loss = sparkgd.step(dataset, model, grad, desc, dataset_count)
    print loss

print 'Validating negatives'
negs = 0
for X in neg:
    pred = model['w'].dot(X) + model['b']
    if pred < 0:
        negs += 1
    print pred
print 'Validating positives'
poss = 0
for X in pos:
    pred = model['w'].dot(X) + model['b']
    if pred > 0:
        poss += 1
    print pred
print negs, poss
