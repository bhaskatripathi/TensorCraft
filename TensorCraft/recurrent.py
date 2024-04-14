from .unit import *
from .dense import *
import tensorflow as tf


class BasicRNNCell(Unit, tf.contrib.rnn.RNNCell):

    def __init__(self, recurrent_layer):
        super().__init__()
        self._recurrent_layer = recurrent_layer
        self.register_subunit(recurrent_layer)
        self._state_size = self._recurrent_layer.output_dim

    @property
    def recurrent_layer(self):
        return self._recurrent_layer

    @classmethod
    def from_description(cls, n_in, n_hid, act, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn"):
            recurrent_layer = DenseLayer.from_description(n_in+n_hid, n_hid, act, scope)
        return cls(recurrent_layer)

    @property
    def state_size(self):
        return (self.recurrent_layer.output_dim,)

    @property
    def output_size(self):
        return self.recurrent_layer.output_dim

    def process(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn_output") as scope:
            outputs = self.recurrent_layer.process(tf.concat([inputs, state[0]], 1), scope)
        return outputs, (outputs,)

    def to_dictionary(self, session):
        return {"recurrent_layer": self.recurrent_layer.to_dictionary(session)}

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn"):
            recurrent_layer = DenseLayer.from_dictionary(data_dict["recurrent_layer"], scope)
        return cls(recurrent_layer)

pass


class BasicRNN(StatefulUnit):

    def __init__(self, cell, hid_state=None):
        super().__init__(cell, (hid_state,))

    @classmethod
    def from_cell(cls, cell, n_states=None, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn"):
            hid_state = tf.get_variable(
                "hidden_state",
                shape=[n_states, cell.state_size[0]],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=False
            ) if n_states is not None else None
        return cls(cell, hid_state)

    @property
    def hidden_state(self):
        return self.states[0]

    def to_dictionary(self, session):
        return {
            "cell": self.cell.to_dictionary(session),
            "hiddenstate": session.run(self.hidden_state) if self.hidden_state is not None else "none"
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, default_name="basic_rnn"):
            cell = BasicRNNCell.from_dictionary(data_dict["cell"])
            hid_state = tf.get_variable(
                "hidden_state",
                shape=data_dict["hiddenstate"].shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(data_dict["hiddenstate"]),
                trainable=False
            ) if not isinstance(data_dict["hiddenstate"], str) else None
        return cls(cell, hid_state)


pass


class LSTMCell(Unit, tf.contrib.rnn.RNNCell):

    def __init__(self, forget_gate, input_gate, input_extractor, output_gate, output_extractor):
        super().__init__()
        self._forget_gate = forget_gate
        self._input_gate = input_gate
        self._input_extractor = input_extractor
        self._output_gate = output_gate
        self._output_extractor = output_extractor
        self.register_subunit(forget_gate)
        self.register_subunit(input_gate)
        self.register_subunit(input_extractor)
        self.register_subunit(output_gate)
        self.register_subunit(output_extractor)
        self._state_size = output_extractor.output_dim, input_extractor.output_dim

    @classmethod
    def from_description(cls, n_in, n_hid, n_cell, scope=None):
        with tf.variable_scope(scope, "lstm_cell"):
            forget_gate = DenseLayer.from_description(n_in + n_hid, n_cell, None, "forget_gate")
            input_gate = DenseLayer.from_description(n_in + n_hid, n_cell, tf.sigmoid, "input_gate")
            input_extractor = DenseLayer.from_description(n_in + n_hid, n_cell, tf.tanh, "input_extractor")
            output_gate = DenseLayer.from_description(n_in + n_hid, n_hid, tf.sigmoid, "output_gate")
            output_extractor = DenseLayer.from_description(n_cell, n_hid, tf.tanh, "output_extractor")
        return cls(forget_gate, input_gate, input_extractor, output_gate, output_extractor)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self.output_extractor.output_dim

    @property
    def forget_gate(self):
        return self._forget_gate

    @property
    def input_gate(self):
        return self._input_gate

    @property
    def input_extractor(self):
        return self._input_extractor

    @property
    def output_gate(self):
        return self._output_gate

    @property
    def output_extractor(self):
        return self._output_extractor

    def process(self, inputs, state, scope=None):
        with tf.variable_scope(scope, "lstm_cell_output"):
            hidden_state, cell_state = state
            hidden_inputs = tf.concat([inputs, hidden_state], axis=1)
            cell_state = cell_state * tf.sigmoid((self.forget_gate.process(hidden_inputs) + 1))
            cell_state += self.input_gate.process(hidden_inputs) * self.input_extractor.process(hidden_inputs)
            hidden_state = self.output_gate.process(hidden_inputs) * self.output_extractor.process(cell_state)
        return hidden_state, (hidden_state, cell_state)

    def to_dictionary(self, session):
        return {
            "forget_gate": self.forget_gate.to_dictionary(session),
            "input_gate": self.input_gate.to_dictionary(session),
            "input_extractor": self.input_extractor.to_dictionary(session),
            "output_gate": self.output_gate.to_dictionary(session),
            "output_extractor": self.output_extractor.to_dictionary(session)
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, "lstm_cell"):
            forget_gate = DenseLayer.from_dictionary(data_dict["forget_gate"], "forget_gate")
            input_gate = DenseLayer.from_dictionary(data_dict["input_gate"], "input_gate")
            input_extractor = DenseLayer.from_dictionary(data_dict["input_extractor"], "input_extractor")
            output_gate = DenseLayer.from_dictionary(data_dict["output_gate"], "output_gate")
            output_extractor = DenseLayer.from_dictionary(data_dict["output_extractor"], "output_extractor")
        return cls(forget_gate, input_gate, input_extractor, output_gate, output_extractor)


pass


class LSTM(StatefulUnit):

    def __init__(self, cell, hid_state, cell_state):
        super().__init__(cell, (hid_state, cell_state))

    @classmethod
    def from_cell(cls, cell, n_states=None, scope=None):
        with tf.variable_scope(scope, default_name="lstm"):
            n_hid, n_cell = cell.state_size
            init = tf.zeros_initializer()
            hid_state = tf.get_variable(
                "hidden_state",
                shape = [n_states, n_hid],
                initializer = init,
                trainable = False
            ) if n_states is not None else None
            cell_state = tf.get_variable(
                "cell_state",
                shape = [n_states, n_cell],
                initializer = init,
                trainable = False
            ) if n_states is not None else None
        return cls(cell, hid_state, cell_state)

    @property
    def hidden_state(self):
        return self.states[0]

    @property
    def cell_state(self):
        return self.states[1]

    def to_dictionary(self, session):
        return {
            "cell": self.cell.to_dictionary(session),
            "hiddenstate": session.run(self.hidden_state) if self.hidden_state is not None else "none",
            "cellstate": session.run(self.cell_state) if self.cell_state is not None else "none"
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, default_name="lstm"):
            cell = LSTMCell.from_dictionary(data_dict["cell"], "lstm_cell")
            hid_state = tf.get_variable(
                "hidden_state",
                shape=data_dict["hiddenstate"].shape,
                initializer=tf.constant_initializer(data_dict["hiddenstate"]),
                trainable=False
            ) if not isinstance(data_dict["hiddenstate"], str) else None
            cell_state = tf.get_variable(
                "cell_state",
                shape=data_dict["cellstate"].shape,
                initializer=tf.constant_initializer(data_dict["cellstate"]),
                trainable=False
            ) if not isinstance(data_dict["hiddenstate"], str) else None
            lstm = cls(cell, hid_state, cell_state)
        return lstm

pass


class BidirectionalRNN(Unit):

    def __init__(self, forward_rnn, backward_rnn):
        super().__init__()
        self._forward_rnn = forward_rnn
        self._backward_rnn = backward_rnn
        self.register_subunit(self._forward_rnn)
        self.register_subunit(self._backward_rnn)

    @property
    def forward_rnn(self):
        return self._forward_rnn

    @property
    def backward_rnn(self):
        return self._backward_rnn

    def process(self, inputs, merge_outputs=True, scope=None, **kwargs):
        with tf.variable_scope(scope, default_name="bidirectional_rnn_output"):
            fwd_outputs = self.forward_rnn.process(inputs, include_state=False, **kwargs)
            bwd_inputs = tf.reverse(inputs, axis=[1])
            bwd_bwd_outputs = self.backward_rnn.process(bwd_inputs, include_state=False, **kwargs)
            bwd_outputs = tf.reverse(bwd_bwd_outputs, axis=[1])
            outputs = tf.concat([fwd_outputs, bwd_outputs], axis=-1) if merge_outputs else (fwd_outputs, bwd_outputs)
        return outputs

    def to_dictionary(self, session):
        return {
            "forward_rnn": {
                "type": type(self.forward_rnn).__name__,
                "rnn": self.forward_rnn.to_dictionary(session)
            },
            "backward_rnn": {
                "type": type(self.backward_rnn).__name__,
                "rnn": self.backward_rnn.to_dictionary(session)
            }
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        fwd_rnn_dict = data_dict["forward_rnn"]
        fwd_rnn_cls_string = fwd_rnn_dict["type"]
        fwd_rnn_cls = Unit.class_by_name(fwd_rnn_cls_string)
        fwd_rnn_data_dict = fwd_rnn_dict["rnn"]
        bwd_rnn_dict = data_dict["backward_rnn"]
        bwd_rnn_cls_string = bwd_rnn_dict["type"]
        bwd_rnn_cls = Unit.class_by_name(bwd_rnn_cls_string)
        bwd_rnn_data_dict = bwd_rnn_dict["rnn"]
        with tf.variable_scope(scope, default_name="bidirectional_rnn"):
            forward_rnn = fwd_rnn_cls.from_dictionary(fwd_rnn_data_dict)
            backward_rnn = bwd_rnn_cls.from_dictionary(bwd_rnn_data_dict)
        return cls(forward_rnn, backward_rnn)

pass