import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class GenMLP(Chain):
    def __init__(self, dim_z, n_units, n_out):
        super(GenMLP, self).__init__(
            l1=L.Linear(dim_z, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_units),
            l4=L.Linear(n_units, n_out),
            bn1=L.BatchNormalization(n_units),
            bn2=L.BatchNormalization(n_units),
            bn3=L.BatchNormalization(n_units),
        )

    def __call__(self, z, test=False):
        h = F.relu(self.bn1(self.l1(z), test=test))
        h = F.relu(self.bn2(self.l2(h), test=test))
        h = F.relu(self.bn3(self.l3(h), test=test))
        return F.sigmoid(self.l4(h))


class DisMLP(Chain):
    def __init__(self, n_in, n_units):
        super(DisMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, 1),
        )

    def __call__(self, x, train=True):
        x = F.dropout(x, ratio=0.1, train=train)
        h = F.dropout(F.relu(self.l1(x)), ratio=0.5, train=train)
        h = F.dropout(F.relu(self.l2(h)), ratio=0.5, train=train)
        return self.l3(h)


class GANs:
    def __init__(self, generator, discriminator, dim_data, dim_z):
        self.generator = generator
        self.discriminator = discriminator
        self.dim_data = dim_data
        self.dim_z = dim_z
        self.setup_optimizers()

    def dis_loss(self, x, t):
        return F.sigmoid_cross_entropy(self.discriminator(x), t)

    def gen_loss(self, z):
        return F.sigmoid_cross_entropy(
            self.discriminator(self.generator(z), train=False),
            chainer.Variable(np.ones((z.shape[0], 1), dtype=np.int32))
        )

    def setup_optimizers(self):
        self.gen_optimizer = optimizers.Adam(alpha=1e-5)
        self.gen_optimizer.use_cleargrads()
        self.gen_optimizer.setup(self.generator)
        self.dis_optimizer = optimizers.Adam(alpha=1e-5)
        self.dis_optimizer.use_cleargrads()
        self.dis_optimizer.setup(self.discriminator)

    def train(self, dataset, n_epochs, batch_size, report=None, z_test=None):
        label0s = chainer.Variable(np.zeros((batch_size, 1), dtype=np.int32))
        label1s = chainer.Variable(np.ones((batch_size, 1), dtype=np.int32))
        # Epochs
        for epoch in range(1, n_epochs+1):
            print("Initiating an epoch #", epoch, flush=True)
            perm = np.random.permutation(dataset.shape[0])

            dis_epochloss = 0.0
            gen_epochloss = 0.0

            for i in range(0, dataset.shape[0], batch_size):
                # Process a mini batch
                data_batch = chainer.Variable(dataset[perm[i:i+batch_size]])
                # Update discriminator
                z_batch = chainer.Variable(self.sample_z(batch_size))
                dis_batchloss = self.dis_loss(data_batch, label1s)
                dis_batchloss = dis_batchloss + self.dis_loss(
                    self.generator(z_batch), label0s
                )
                self.discriminator.cleargrads()
                dis_batchloss.backward()
                self.dis_optimizer.update()
                dis_epochloss = dis_epochloss + dis_batchloss.data

                # Update generator
                z_batch = chainer.Variable(self.sample_z(batch_size))
                gen_batchloss = self.gen_loss(z_batch)
                self.generator.cleargrads()
                gen_batchloss.backward()
                self.gen_optimizer.update()
                gen_epochloss = gen_epochloss + gen_batchloss.data
            print("    Discriminator loss in a epoch :", dis_epochloss)
            print("    Generator loss in a epoch     :", gen_epochloss)
            print("", flush=True)
            if (epoch%5 == 0) and (report is not None) and (z_test is not None):
                report(self.generator, epoch, z_test)

    def sample_z(self, n_samples):
        return np.random.uniform(-1, 1, (n_samples, self.dim_z)).astype(np.float32)


def make_dataset():
    # MNIST dataset without labels
    train, test = datasets.get_mnist(withlabel=False)
    return np.vstack((train, test,))

def update_net(loss, optimizer, arg1, arg2):
    loss.cleargrads()
    batchloss = loss(arg1, arg2)
    batchloss.backward()
    optimizer.update()
    return batchloss

def save_digits(generator, epoch, z_test):
    fig = plt.figure()
    samples = generator(z_test).data
    samples = [
        sample.reshape((28, 28)) for sample in samples
    ]
    for i, sample in zip(range(1, z_test.shape[0]+1), samples):
        ax = fig.add_subplot(1, z_test.shape[0], i)
        ax.matshow(sample, cmap=mpl.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
#    plt.show()
    plt.savefig(str(epoch)+".pdf")


def main():
    dim_data = 784

    # Hyperparameters
#    dim_z = 10
#    gen_hidden, dis_hidden = 5, 5
    dim_z = 100
    gen_hidden, dis_hidden = 1000, 1000

    batch_size = 50
    n_epochs = 50

    generator = GenMLP(dim_z, gen_hidden, dim_data)
    discriminator = DisMLP(dim_data, dis_hidden)

    gans = GANs(generator, discriminator, dim_data, dim_z)

    dataset = make_dataset()
    gans.train(
        dataset, n_epochs, batch_size,
        report=save_digits, z_test=chainer.Variable(gans.sample_z(10))
    )

    serializers.save_npz("generator.npz", gans.generator)
    serializers.save_npz("discriminator.npz", gans.discriminator)

if __name__ == '__main__':
    main()
