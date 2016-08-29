import theano as T
import theano.tensor as TT
import theano.tensor.nnet as NN
from theano.compile.nanguardmode import NanGuardMode

import h5py
import numpy as NP
import numpy.random as RNG

from collections import OrderedDict

dataset = h5py.File("mnist.h5")

train_set = dataset["train/inputs"][:50000]
valid_set = dataset["train/inputs"][50000:]
test_set = dataset["test/inputs"]

dim0 = 300
dim1 = 100
dim2 = 60
dim_latent = 2

batch_size = 50

lr = 0.001
rho = 0.9
epsilon = 1e-6

def glorot(shape):
    fan_in = shape[0]
    fan_out = shape[1]
    s = NP.sqrt(6. / (fan_in + fan_out))
    return RNG.uniform(low=-s, high=s, size=shape)

WE0 = T.shared(glorot((28 * 28, dim0)))
BE0 = T.shared(NP.zeros((dim0,)))
WE3 = T.shared(glorot((dim0, dim_latent * 2)))
BE3 = T.shared(NP.zeros((dim_latent * 2,)))

WD3 = T.shared(glorot((dim_latent, dim0)))
BD3 = T.shared(NP.zeros((dim0,)))
WD0 = T.shared(glorot((dim0, 28 * 28)))
BD0 = T.shared(NP.zeros((28 * 28,)))

params = [WE0, BE0, WE3, BE3, WD3, BD3, WD0, BD0]

x = TT.tensor3('x')
eps = TT.matrix('eps')
x_ = TT.flatten(x, outdim=2) * 2 - 1
e0 = TT.tanh(TT.dot(x_, WE0) + BE0)
enc_params = TT.dot(e0, WE3) + BE3
enc_mu = enc_params[:, :dim_latent]
enc_log_sigma = enc_params[:, dim_latent:]
enc_sigma_sqr = TT.exp(enc_log_sigma)
enc = enc_mu + TT.sqrt(enc_sigma_sqr) * eps
d0 = TT.tanh(TT.dot(enc, WD3) + BD3)
y_ = TT.clip(NN.sigmoid(TT.dot(d0, WD0) + BD0), 1e-6, 1 - 1e-6)
_y = TT.reshape(y_, (-1, 28, 28))

x_flat = x.flatten()
y_flat = _y.flatten()

rec_cost = -(x * TT.log(_y) + (1 - x) * TT.log(1 - _y)).sum() / x.shape[0]
reg_cost = 0.5 * (enc_sigma_sqr.sum(axis=1) + (enc_mu ** 2).sum(axis=1) - dim_latent - TT.log(enc_sigma_sqr).sum(axis=1)).sum() / x.shape[0]
cost = rec_cost + reg_cost

grads = T.grad(cost, params)

updates = OrderedDict()
acc = [T.shared(NP.zeros(p.get_value().shape, dtype=T.config.floatX)) for p in params]
grads_norm = TT.sqrt(sum(map(lambda x: TT.sqr(x).sum(), grads)))
grads_max = TT.max([TT.max(g) for g in grads])
for p, g, a in zip(params, grads, acc):
    g = TT.clip(g, -0.1, 0.1)
    #g = TT.switch(grads_norm > 5, g / grads_norm * 5, g)
    new_a = rho * a + (1 - rho) * g ** 2
    updates[a] = new_a
    new_p = p - lr * g / TT.sqrt(new_a + epsilon)
    updates[p] = new_p

test = T.function([x, eps], [_y, cost])
train = T.function([x, eps], [_y, rec_cost, reg_cost, cost, grads_norm, grads_max], updates=updates, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
epoch = 0
prev_avg_cost_valid = 100

try:
	while True:
		avg_cost_valid = 0
		avg_cost_train = 0
		for i in range(0, 50000 / batch_size):
			train_batch = RNG.randint(0, 50000, batch_size)
			train_x = train_set[train_batch] / 255.
			train_eps = RNG.normal(0, 1, (batch_size, dim_latent))
			valid_sample = RNG.randint(0, 10000)
			valid_x = [valid_set[valid_sample] / 255.]
			valid_eps = RNG.normal(0, 1, (1, dim_latent))
			_, cost_valid = test(valid_x, valid_eps)
			_, rec_cost, reg_cost, cost_train, gradn, gradm = train(train_x, train_eps)
			avg_cost_valid = (avg_cost_valid * i + cost_valid) / (i + 1)
			avg_cost_train = (avg_cost_train * i + cost_train) / (i + 1)
                print 'Epoch #%d Train %+0.10f Valid %+0.10f Rec %+0.10f Reg %+0.10f' % (epoch, avg_cost_train, avg_cost_valid, rec_cost, reg_cost)
		epoch += 1
except KeyboardInterrupt:
	pass

m = h5py.File("model-v.h5")
m["WE0"] = WE0.get_value()
m["BE0"] = BE0.get_value()
m["WE3"] = WE3.get_value()
m["BE3"] = BE3.get_value()
m["WD0"] = WD0.get_value()
m["BD0"] = BD0.get_value()
m["WD3"] = WD3.get_value()
m["BD3"] = BD3.get_value()
m.close()
