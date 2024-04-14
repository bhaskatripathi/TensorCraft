from .unit import *
import tensorflow as tf


class Reshaper(Unit):

    def __init__(self, target_shape):
        super().__init__()
        self._target_shape = target_shape
        self._output_dim = self._target_shape[-1]

    @property
    def target_shape(self):
        return self._target_shape

    @property
    def output_dim(self):
        return self._output_dim

    def process(self, inputs, scope=None):
        with tf.variable_scope(scope, default_name="reshape"):
            output = tf.reshape(inputs, shape=self.target_shape)
        return output

pass


class InvertibleReshaper(Reshaper):

    def __init__(self, target_shape, original_shape):
        super().__init__(target_shape)
        self._inverse = Reshaper(original_shape)

    @property
    def inverse(self):
        return self._inverse

pass


class BatchFlattener(InvertibleReshaper):

    def __init__(self, batch_shape):
        super().__init__([-1] + list(batch_shape[-1:]), batch_shape)
        self._batch_shape = list(batch_shape[:-1]) + [-1]

    pass


class Flattener(InvertibleReshaper):

    def __init__(self, original_shape):
        super().__init__([-1], original_shape)

pass