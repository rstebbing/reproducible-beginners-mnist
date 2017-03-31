"""An adaption of MNIST For ML Beginners that includes the options to seed the
global random state, save and load model state, and also use the
`SimpleBatchFeeder` to continue model training consistently from one execution
to the next."""
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from batch_feeder import SimpleBatchFeeder


def main():
    parser = ArgumentParser()
    parser.add_argument('train_dir', nargs='?', default='MNIST_data')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.5)
    parser.add_argument('-n', '--num-batches', type=int, default=1000)
    parser.add_argument('-b', '--batch-size', type=int, default=100)
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('--restore')
    parser.add_argument('--save')
    parser.add_argument('--batch-feeder-start', type=int)
    args = parser.parse_args()

    # Read the input dataset, downloading it if necessary to `train_dir`.
    print('train_dir: {}'.format(args.train_dir))
    mnist = input_data.read_data_sets(args.train_dir, one_hot=True)
    num_train, D = mnist.train.images.shape
    num_train_labels, L = mnist.train.labels.shape
    assert num_train == num_train_labels
    print('num_train: {}, D: {}, L: {}'.format(num_train, D, L))

    # Prepare the computation graph all the way to `train_step` (which updates
    # the variables `W` and `b`).
    W = tf.Variable(tf.zeros([D, L]))
    b = tf.Variable(tf.zeros([L]))

    x = tf.placeholder(tf.float32, (None, D))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, (None, L))
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    # Define `next_batch`, a zero-argument callable that returns a nested tuple
    # of the form `(next_start, (batch_x, batch_y))`:
    # - `next_start` is the value to specify for `--batch-feeder-start` so
    #   that the model training can continue as if it had not been interrupted,
    #   given: the same `--batch-size`, `--seed`, and saved model state (via
    #   `--restore`).
    # - `batch_x` and `batch_y` are the batch of images and labels.
    print('batch_feeder_start: {}, seed: {}'.format(
        args.batch_feeder_start, args.seed))
    if args.batch_feeder_start is None:
        if args.seed is not None:
            np.random.seed(args.seed)

        def next_batch():
            return None, mnist.train.next_batch(args.batch_size)
    else:
        batch_feeder = SimpleBatchFeeder(
            mnist.train.images, mnist.train.labels,
            batch_size=args.batch_size, start=args.batch_feeder_start,
            seed=args.seed)
        next_batch = batch_feeder.next_batch

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # Restore the model from `restore` (if specified).
        if args.restore:
            print('restore: {}'.format(args.restore))
            saver = tf.train.Saver()
            saver.restore(sess, args.restore)

        # Run `num_batches` of gradient descent steps, updating
        # `next_start` as each batch is processed.
        next_start = None
        for _ in range(args.num_batches):
            next_start, (batch_x, batch_y) = next_batch()
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        if next_start is not None:
            print('next_start: {}'.format(next_start))

        # Finally, save the model variables (if `save` has been specified).
        if args.save:
            print('save: {}'.format(args.save))

            # (Ensure the directory specified by `save` exists.)
            save_dir = os.path.dirname(args.save)
            try:
                os.makedirs(save_dir)
            except OSError:
                if not os.path.isdir(save_dir):
                    raise

            saver = tf.train.Saver()
            saver.save(sess, args.save)

        # Evaluate the model on the complete test set of images and labels ...
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        evaluated_accuracy = sess.run(accuracy,
                                      feed_dict={x: mnist.test.images,
                                                 y_: mnist.test.labels})
        # ... and report the evaluated accuracy.
        print('accuracy: {}'.format(evaluated_accuracy))


if __name__ == '__main__':
    main()
