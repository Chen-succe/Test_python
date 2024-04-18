import numpy as np


class Node:
    def __init__(self, name, inputs=None, is_trainable=False):
        if inputs is None:
            inputs = []
        self.name = name
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradient = dict()  # loss 对于字典中每个keys的偏导
        self.is_trainable = is_trainable

        for node in self.inputs:
            node.outputs.append(self)

    def forward(self):
        pass

    def backward(self):
        pass

    def __repr__(self):
        return self.name


class Placeholer(Node):
    def __init__(self, name, is_trainable=True):
        Node.__init__(self, name=name, is_trainable=is_trainable)

    def forward(self):
        pass

    def backward(self):
        self.gradient[self] = self.outputs[0].gradient[self]

    def __repr__(self):
        return 'Placeholer:{}'.format(self.name)


class Linear(Node):
    def __init__(self, name, inputs=None):
        Node.__init__(self, name=name, inputs=inputs)
        if inputs is None:
            inputs = []

    def forward(self):
        x, k, b = self.inputs[0], self.inputs[1], self.inputs[2]
        self.value = k.value * x.value + b.value

    def backward(self):
        x, k, b = self.inputs[0], self.inputs[1], self.inputs[2]
        self.gradient[k] = self.outputs[0].gradient[self] * x.value
        self.gradient[b] = self.outputs[0].gradient[self] * 1
        self.gradient[x] = self.outputs[0].gradient[self] * k.value

    def __repr__(self):
        return 'Linear:{}'.format(self.name)


class Sigmoid(Node):
    def __init__(self, name, inputs=None):
        Node.__init__(self, name=name, inputs=inputs)
        if inputs is None:
            inputs = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        x = self.inputs[0]
        self.value = self._sigmoid(x.value)

    def backward(self):
        x = self.inputs[0]
        self.gradient[x] = self.outputs[0].gradient[self] * self._sigmoid(x.value) * (1 - self._sigmoid(x.value))

    def __repr__(self):
        return 'Sigmoid:{}'.format(self.name)


class Loss(Node):
    def __init__(self, name, inputs=None):
        Node.__init__(self, name=name, inputs=inputs)
        if inputs is None:
            inputs = []

    def forward(self):
        y = self.inputs[0]
        yhat = self.inputs[1]
        self.value = np.mean((y.value - yhat.value) ** 2)

    def backward(self):
        y = self.inputs[0]
        yhat = self.inputs[1]
        self.gradient[y] = 2 * np.mean((y.value - yhat.value))
        self.gradient[yhat] = -2 * np.mean((y.value - yhat.value))

    def __repr__(self):
        return 'Linear:{}'.format(self.name)


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, y, a)

    def forward(self):
        y = self.inputs[0].value.reshape(-1, 1)
        a = self.inputs[1].value.reshape(-1, 1)
        assert (y.shape == a.shape)

        self.m = self.inputs[0].value.shape[0]
        self.diff = y - a
        self.value = np.mean(self.diff ** 2)

    def backward(self):
        self.gradient[self.inputs[0]] = (2 / self.m) * self.diff
        self.gradient[self.inputs[1]] = (-2 / self.m) * self.diff