import TensorCraft as fg
import tensorflow as tf
import gzip
import numpy as np
import argparse
import os


train_imgs_file = "train-images-idx3-ubyte.gz"
train_lbls_file = "train-labels-idx1-ubyte.gz"
valid_imgs_file = "t10k-images-idx3-ubyte.gz"
valid_lbls_file = "t10k-labels-idx1-ubyte.gz"

n_in = 784
n_classes = 10


def read_mnist_images(filename, img_dim=28):
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape((-1, img_dim*img_dim))


def read_mnist_labels(filename):
    with gzip.open(filename, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels


def read_mnist(imgs_file, lbls_file, img_dim=28):
    imgs = read_mnist_images(imgs_file, img_dim)
    lbls = read_mnist_labels(lbls_file)
    return(imgs, lbls)


def create_dataset_variables(imgs_array, lbls_array, scope=None):
    with tf.variable_scope(scope, default_name="mnist"):
        imgs = tf.constant(np.asarray(imgs_array, np.float32), dtype=tf.float32)
        lbls = tf.constant(np.asarray(lbls_array, np.int32), dtype=tf.int32)
    return (imgs, lbls)


def create_dataset(imgs_array, lbls_array, scope=None):
    imgs, lbls = create_dataset_variables(imgs_array, lbls_array, scope)
    return tf.data.Dataset.from_tensor_slices((imgs, lbls))


def loss(logits, labels, scope=None):
    with tf.variable_scope(scope, default_name="loss"):
        loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss_vector)
    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", "-i", default=os.getcwd())
    parser.add_argument("--save-directory", "-o", default=os.getcwd())
    parser.add_argument("--layer-sizes", "-p", nargs="+", default=[], type=int)
    parser.add_argument("--batch-size", "-bs", default=64, type=int)
    parser.add_argument("--number-epochs", "-n", default=5, type=int)
    parser.add_argument("--seed", "-s", default=123, type=int)
    parser.add_argument("--learning-rate", "-lr", default=.02, type=float)
    args = parser.parse_args()

    train_imgs_path = os.path.join(args.data_directory, train_imgs_file)
    train_lbls_path = os.path.join(args.data_directory, train_lbls_file)
    train_imgs_array, train_lbls_array = read_mnist(train_imgs_path, train_lbls_path)
    valid_imgs_path = os.path.join(args.data_directory, valid_imgs_file)
    valid_lbls_path = os.path.join(args.data_directory, valid_lbls_file)
    valid_imgs_array, valid_lbls_array = read_mnist(valid_imgs_path, valid_lbls_path)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        # data
        trainset = create_dataset(train_imgs_array, train_lbls_array).batch(args.batch_size)
        valid_imgs, valid_lbls = create_dataset_variables(valid_imgs_array, valid_lbls_array)
        it = trainset.make_initializable_iterator()
        train_imgs, train_lbls = it.get_next()
        # model
        model = fg.DenseClassifier.from_description(n_in, n_classes, args.layer_sizes, [tf.tanh]*len(args.layer_sizes))
        train_logits = model.process(train_imgs)
        valid_logits = model.process(valid_imgs)
        valid_preds = model.predict(valid_imgs)
        # training
        train_loss = loss(train_logits, train_lbls)
        train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(train_loss, var_list=model.parameters)
        # validation
        valid_loss = loss(valid_logits, valid_lbls)
        valid_acc = tf.reduce_mean(tf.cast(tf.equal(valid_preds, valid_lbls), tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(it.initializer)
            epoch = 1
            train_loss_epoch_hist = []
            while epoch <= args.number_epochs:
                try:
                    _, tl = sess.run([train_step, train_loss])
                    train_loss_epoch_hist.append(tl)
                except tf.errors.OutOfRangeError:
                    sess.run(it.initializer)
                    tel = np.mean(train_loss_epoch_hist)
                    train_loss_epoch_hist = []
                    vl, vp, va = sess.run([valid_loss, valid_preds, valid_acc])
                    print("epoch %i: loss %f (train) / %f (valid), acc. %.3f%%" % (epoch, tel, vl, va * 100), end=".\n")
                    epoch += 1
            model.save(sess, os.path.join(args.save_directory, "dense_mnist.hdf5"))