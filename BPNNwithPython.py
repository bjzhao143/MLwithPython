import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# load mnist dataset
data = mnist.load_data()[1]
X ,y = data[0].reshape((10000,-1)),data[1].reshape((-1,1))

ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()
X_train, X_val, y_train, y_val = train_test_split(X, y,shuffle=True)

# Model
class NN(object):
    def __init__(self, n_inputs, n_hiddens, n_outputs, lr=0.001):
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.lr = lr
        self.w_h = np.random.randn(self.n_inputs, self.n_hiddens)
        self.b_h = np.zeros((1, self.n_hiddens))
        self.w_o = np.random.randn(self.n_hiddens, n_outputs)
        self.b_o = np.zeros((1, self.n_outputs))

    def sigmoid(self, X):
        return 1.0 / (1 + np.exp(-X))

    def forward(self, X):
        A_h = np.dot(X, self.w_h) + self.b_h
        o_h = self.sigmoid(A_h)
        A_o = np.dot(o_h, self.w_o) + self.b_o
        o_o = self.sigmoid(A_o)
        outputs = {
            'A_h': A_h,
            'o_h': o_h,
            'A_o': A_o,
            'o_o': o_o
        }
        return outputs

    def mse_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def back_propagation(self, X, y, outputs):
        g = outputs['o_o'] * (1 - outputs['o_o']) * (outputs['o_o'] - y)
        dw_o = np.dot(outputs['o_h'].T, g)
        db_o = np.sum(g)
        e = np.dot(g, self.w_o.T) * outputs['o_h'] * (1 - outputs['o_h'])
        dw_h = np.dot(X.T, e)
        db_h = np.sum(e)
        gradients = {
            'dw_o': dw_o,
            'db_o': db_o,
            'dw_h': dw_h,
            'db_h': db_h
        }
        return gradients

    def update_weights(self, gradients):
        self.w_o -= self.lr * gradients['dw_o']
        self.b_o -= self.lr * gradients['db_o']
        self.w_h -= self.lr * gradients['dw_h']
        self.b_h -= self.lr * gradients['db_h']

    def val_test(self, X_val, y_val):
        outputs = self.forward(X_val)
        val_loss = self.mse_loss(y_val, outputs['o_o'])
        return val_loss

    def train(self, X_train, y_train, X_val, y_val, n_iters=1000):
        for epoch in range(n_iters):
            outputs = self.forward(X)
            loss = self.mse_loss(y, outputs['o_o'])
            val_loss = self.val_test(X_val, y_val)
            gradients = self.back_propagation(X, y, outputs)
            self.update_weights(gradients)
            if epoch % 10 == 0:
                print('epoch:%d,train_loss:%s val_loss:%s' % (epoch, loss, val_loss))

    def predict(self, x):
        outputs = self.forward(x)
        y_pred = np.array([np.argmax(i) for i in outputs['o_o']])
        return y_pred

# Train
nn = NN(784,100,10)
nn.train(X_train,y_train,X_val,y_val,n_iters=200)