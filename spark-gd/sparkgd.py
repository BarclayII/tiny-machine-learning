
import pyspark
import numpy as NP
import copy
import numbers

class MeanAccumulatorParam(pyspark.AccumulatorParam):
    def addInPlace(self, v1, v2):
        v1[1] = (v1[1] * v1[0] + v2[1] * v2[0]) / (v1[0] + v2[0])
        v1[0] += v2[0]
        return v1

class MeanLossAccumulatorParam(MeanAccumulatorParam):
    def zero(self, initialValue):
        return [0, 0.]

class MeanGradAccumulatorParam(MeanAccumulatorParam):
    def zero(self, initialValue):
        return [0, NP.zeros_like(initialValue[1], dtype='float32')]


def _get_samples(df, df_count, size):
    # http://stackoverflow.com/questions/34003314/how-take-a-random-row-from-a-pyspark-dataframe
    fraction_bound = max(size * 1.5, 100.) / df_count
    return df.sample(False, fraction_bound).limit(size)


def train_model(model, train_set, valid_set, eval_, grad, desc,
                batch_size=256, times_per_train_epoch=None,
                times_per_valid_epoch=None, max_epochs=None,
                patience=50, patience_increase=2,
                improvement_threshold=0.995):
    '''
    model : dict of {param_name: param_value}
    train_set, valid_set : DataFrame
    eval_ : func (data_point, model) -> (metric) (the smaller the better)
    grad : func (data_point, model) -> (loss, {param_name: param_grad})
    desc : func (model, {param_name: param_grad}) -> ()
    '''
    train_count = train_set.count()
    valid_count = valid_set.count()

    if times_per_train_epoch is None:
        times_per_train_epoch = train_count / batch_size
    if times_per_valid_epoch is None:
        times_per_valid_epoch = valid_count / batch_size
    if max_epochs is None:
        max_epochs = NP.inf

    sc = train_set.rdd.context

    # http://deeplearning.net/tutorial/gettingstarted.html#early-stopping
    epoch = 0
    best_model = None
    best_valid_loss = NP.inf

    while epoch < max_epochs:
        epoch += 1

        for batch in range(times_per_train_epoch):
            train_batch = (_get_samples(train_set, train_count, batch_size)
                           .repartition(batch_size))

            model_bc = {c: sc.broadcast(model[c]) for c in model}
            loss_acc = sc.accumulator([0, 0.], MeanLossAccumulatorParam())
            grad_acc = {c: sc.accumulator([0, NP.zeros_like(model[c], dtype='float32')],
                                          MeanGradAccumulatorParam())
                        for c in model}

            def _accum_losses(r, loss_acc=loss_acc, grad_acc=grad_acc,
                              model_bc=model_bc):
                loss, grads = grad(r, {c: model_bc[c].value for c in model})
                print ('Accumulating loss %f' % loss)
                loss_acc.add([1, loss])
                for c in grads:
                    print ('Accumulating gradient w.r.t. %s' % c)
                    grad_acc[c].add([1, grads[c]])

            train_batch.foreach(_accum_losses)
            print '#%-8d@%-9d%f' % (epoch, batch, loss_acc.value[1])

            grads = {c: grad_acc[c].value[1] for c in model}
            desc(model, grads)

            for c in model_bc:
                model_bc[c].unpersist()

        valid_loss = 0
        for batch in range(times_per_valid_epoch):
            valid_batch = (_get_samples(valid_set, valid_count, batch_size)
                           .repartition(batch_size))

            model_bc = {c: sc.broadcast(model[c]) for c in model}
            loss_acc = sc.accumulator([0, 0.], MeanLossAccumulatorParam())

            def _accum_losses(r, loss_acc=loss_acc, model_bc=model_bc):
                loss = eval_(r, {c: model_bc[c].value for c in model})
                print ('Accumulating validation loss %f' % loss)
                loss_acc.add([1, loss])

            valid_batch.foreach(_accum_losses)
            valid_loss += loss_acc.value[1]
        valid_loss /= times_per_valid_epoch
        print '#%-8d -> %f' % (epoch, valid_loss)

        if valid_loss < best_valid_loss:
            if valid_loss < best_valid_loss * improvement_threshold:
                patience = max(patience, epoch * patience_increase)
            best_model = copy.deepcopy(model)
            best_valid_loss = valid_loss

        if patience <= epoch:
            print '>%-8d -> %f' % (epoch, valid_loss)
            break
