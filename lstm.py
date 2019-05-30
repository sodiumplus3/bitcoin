import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter, training, datasets, iterators, optimizers, serializers
from chainer.training import extensions
from chainer.datasets import TupleDataset
import math


import matplotlib.pyplot as plt


class lstm(chainer.Chain):
    def __init__(self,n_mid=10,n_out=1):
        super(lstm,self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None,n_mid)
            # self.l1 = L.LSTM(n_mid,n_mid)
            self.l2 = L.LSTM(n_mid,n_mid)
            self.l3 = L.Linear(n_mid,n_out)

    def reset_state(self):
        # self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self,x):
        h = self.l1(x)
        h = self.l2(h)
        h = self.l3(h)
        # h = h[:,-1].reshape(-1,1)
        return h


class LSTM_Classifier(chainer.Chain):
    def __init__(self, predictor,lossfun=F.mean_squared_error):
        super(LSTM_Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.cnt = 0

    def forward(self, x, t):
        self.y = self.predictor(x)
        # print(x,self.y,t)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        self.cnt += 1

        return self.loss

class LSTM_Iterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size=10, seq_len=5, repeat=True):
        self.seq_length = seq_len
        self.dataset = dataset
        self.nsamples = len(dataset)

        self.batch_size = batch_size
        self.repeat = repeat

        self.epoch = 0
        self.iteration = 0
        self.offsets = np.random.randint(0, self.nsamples,size=self.batch_size)

        self.is_new_epoch = False

    def __next__(self):
        if not self.repeat and self.iteration * self.batch_size >= self.nsamples:
            raise StopIteration

        x,t = self.get_data()
        # print(np.array(x).shape,np.array(t).shape)

        self.iteration += 1

        epoch = self.iteration * self.batch_size // self.nsamples

        self.is_new_epoch = self.epoch < epoch

        if self.is_new_epoch:
            self.epoch = epoch
            self.offsets = np.random.randint(0, self.nsamples,size=self.batch_size)
        return list(zip(x, t))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def get_data(self):
        x = []
        for offset in self.offsets:
            tmp = []
            for i in range(self.seq_length):
                tmp.append(self.dataset[(offset + self.iteration - self.seq_length + (i+1)) % len(self.dataset)][0])
            x.append(tmp)

        t = [[self.dataset[(offset + self.iteration + 1) % len(self.dataset)][0]]
                for offset in self.offsets]
        # print(len(x),len(t))
        return x,t

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch     = serializer('epoch', self.epoch)


class LSTM_updater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(LSTM_updater, self).__init__(train_iter, optimizer, device=device)
        self.seq_length = train_iter.seq_length
        self.batch_size = train_iter.batch_size

    def update_core(self):
        loss = 0

        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')


        batch = train_iter.__next__()
        x, t  = self.converter(batch, self.device)
        loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

# データ作成
n_data = 600
sin_data = []
for i in range(n_data+1):
    sin_data.append(math.sin(i/50*math.pi))

# データセット
n_train = 500
n_test  = n_data-n_train


sin_data = np.array(sin_data).astype(np.float32)




x_train, x_test = sin_data[:n_train],sin_data[n_train:]



train = TupleDataset(x_train)
test  = TupleDataset(x_test)


n_seq = 10
net = LSTM_Classifier(lstm(n_mid=n_seq),lossfun=F.mean_squared_error)
# net.compute_accuracy = False
optimizer = optimizers.Adam()
optimizer.setup(net)

train_iter = LSTM_Iterator(train, batch_size = 5, seq_len = n_seq)
test_iter  = LSTM_Iterator(test,  batch_size = 5, seq_len = n_seq, repeat = False)

updater = LSTM_updater(train_iter, optimizer, -1)
trainer = training.Trainer(updater, (30, 'epoch'), out='results/lstm_result')

eval_model = net.copy()
eval_rnn = eval_model.predictor
eval_rnn.train = False
eval_rnn.reset_state()
trainer.extend(extensions.Evaluator(test_iter, eval_model, device=-1), name='val')

trainer.extend(extensions.LogReport(trigger=(1,'epoch'),log_name='log'))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'val/main/loss']))
trainer.extend(training.extensions.PlotReport(['main/loss','val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.dump_graph('main/loss'))
# trainer.extend(extensions.ProgressBar())

trainer.run()
serializers.save_npz('lstm_success.npz',net)




def val1():
    net.predictor.reset_state()
    y = x_train[:n_seq]

    for i in range(n_train):
        new = net.predictor(chainer.Variable(y[-n_seq:].reshape(-1,n_seq)))
        y = np.append(y,new.data[0])

    plt.plot(range(len(y)),y,label='lstm')
    plt.plot(range(n_train),x_train,label='train')
    plt.legend()
    plt.show()

def val2():
    net.predictor.reset_state()
    y = np.array([])
    for i in range(n_seq):
        new = net.predictor(chainer.Variable(x_train[100-n_seq+i:100-n_seq+i+n_seq].reshape(-1,n_seq)))
        y = np.append(y,new.data[-1])
    for i in range(n_train-n_seq):
        new = net.predictor(chainer.Variable(x_train[i:i+n_seq].reshape(-1,n_seq)))
        y = np.append(y,new.data[-1])

    plt.plot(range(len(y)),y,label='lstm')
    plt.plot(range(n_train),x_train,label='train')
    plt.legend()
    plt.show()

val1()
val2()
