# reproducible-beginners-mnist

The purpose of this repository is to provide a very simple example of
structuring a (simple) TensorFlow program so that training on the CPU is (a)
exactly reproducible and (b) can be incrementally continued with the guarantee
that the final model can be _identically_ reproduced with a single execution of
the training program. This is most useful when developing a model, where
repeatability aids debugging and writing tests.

The two most relevant files in the repository are:
1. [train_mnist.py](train_mnist.py), which specifies the main program (similar
   to [mnist_softmax.py][mnist_softmax.py] from
   [MNIST For ML Beginners][MNIST For ML Beginners]).
2. [batch_feeder.py](batch_feeder.py), which implements `SimpleBatchFeeder`.


# Dependencies

This repository has been verified to work under Python 3.5 with TensorFlow 1.1,
NumPy 1.12, and scikit-learn 0.18.1.


# Getting Started

The model generated by [mnist_softmax.py][mnist_softmax.py]
from [MNIST For ML Beginners][MNIST For ML Beginners] is *not* the same from one
execution to the next because NumPy's global random state
(`numpy.random.mtrand._rand`) is used to determine the training baches
([mnist.py L175][mnist.py_L175] and [L189][mnist.py_L189]) but it is *not*
explicitly seeded. The training can be made to be repeatedly simply by calling
`numpy.seed` so that, for a given seed, the training produces the same
model. This is achieved with [train_mnist.py](train_mnist.py) via the `--seed`
command line argument.

For example, to train the simple MNIST classifier using `1000` batches with the
(seeded) _default_ batch feeder run:
```bash
python train_mnist.py --seed 0 \
    --num-batches 1000 \
    --save MNIST_checkpoints/MNIST_0_1000.ckpt
```

The trained classifier achieves a test accuracy of `0.9178` and is identical
from one execution to the next.

Now, _if_ we want to continue training for another `1000` batches we _can_
run [train_mnist.py](train_mnist.py) with `--restore` set to the path of saved
checkpoint so that the model variables are not reinitialized. However, the
resultant classifier will *not* be the same as that generated by training for
`2000` batches from scratch, because the state of the batch feeder is *not*
passed forward from one program execution to the next. This is problematic if we
want to be able to reproduce our model _without_ having to enumerate the
different increments as separate program executions.

The `SimpleBatchFeeder`, which is invoked by specifying the
`--batch-feeder-start` argument, enables continuation of training that _is_
consistent.

For example, to start training using `500` batches from scratch first run:
```bash
python train_mnist.py --seed 0 \
    --num-batches 500 \
    --batch-feeder-start 0 \
    --save MNIST_checkpoints/MNIST_0_500.ckpt
```
Next, to continue training for another `500` batches run:
```bash
python train_mnist.py --seed 0 \
    --num-batches 500 \
    --batch-feeder-start 50000 \
    --restore MNIST_checkpoints/MNIST_0_500.ckpt \
    --save MNIST_checkpoints/MNIST_500_1000.ckpt
```
After the first `500` batches an accuracy of `0.9126` is achieved, and after the
_next_ `500` batches an accuracy of `0.9178` is achieved, *identical* (as
expected) to that achieved by the _single_ run of `1000` batches with the
(seeded) default batch feeder.


# Details

The training of a given model is exactly reproducible when the training program
is deterministic given _all_ inputs, including (but not limited to): the
training data; the model parameters and variable initializations, and the
training parameters (such as batch size and any random state).

The training of a given model can be _consistently_ continued when the training
program is deterministic given _all_ inputs, _and_ the state of the program can
be explicitly passed forward from one execution to the next. In the simple
example given here, given _all_ inputs, the additional state required to
continue consistently is: the model variables `W` and `b`, and the state of the
`SimpleBatchFeeder`.

The model variables are passed forward from one execution to the next via the
checkpoints saved and restored with the `--save` and `--restore` arguments
respectively.

The state of the `SimpleBatchFeeder` is saved _implicitly_ by recording the
total number of rows of training data processed so far, and passing this number
on from one execution to the next. If the `SimpleBatchFeeder` is initialized
with _identical_ training data, parameters, and random state, simply skipping
ahead the same number of rows (using `islice`) is sufficient to restore the
batch feeder to the same state. Crucially, this relies on the underlying
iterator of the batch feeder being fast to evaluate and rerun. In other
scenarios it may be necessary or preferable to _explicitly_ save and load the
state of the batch feeder, in which case the random state and any other
variables should be serialized/deserialized to/from disk accordingly.

[mnist_softmax.py]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
[MNIST For ML Beginners]: https://www.tensorflow.org/get_started/mnist/beginners
[mnist.py_L175]: https://github.com/tensorflow/tensorflow/blob/74c8e5fa8f8131c5cc69037ac51d9667c7c68950/tensorflow/contrib/learn/python/learn/datasets/mnist.py#175
[mnist.py_L189]: https://github.com/tensorflow/tensorflow/blob/74c8e5fa8f8131c5cc69037ac51d9667c7c68950/tensorflow/contrib/learn/python/learn/datasets/mnist.py#189
