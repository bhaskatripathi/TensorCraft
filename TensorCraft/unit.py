import tensorflow as tf
import h5py

class _NameRegistringType(type):

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(cls.mro()) > 2:
            cls._classes_by_name[cls.__name__] = cls

pass

class Unit(object, metaclass=_NameRegistringType):

    _classes_by_name = {}

    @staticmethod
    def class_by_name(name):
        return Unit._classes_by_name[name]

    def __init__(self):
        self._variables = []
        self._parameters = []
        self._subunits = []
        self._initializers = []

    def register_variable(self, variable, register_initializer=True):
        self._variables.append(variable)
        if register_initializer:
            self._initializers.append(variable.initializer)

    def register_parameter(self, parameter, register_initializer=True):
        self.register_variable(parameter, register_initializer)
        self._parameters.append(parameter)

    def register_subunit(self, subunit):
        self._subunits.append(subunit)
        self._initializers += subunit.initializers
        self._variables += subunit.variables
        self._parameters += subunit.parameters

    def register_initializer(self, initializer):
        self._initializers.append(initializer)

    @property
    def variables(self):
        return self._variables

    @property
    def parameters(self):
        return self._parameters

    @property
    def initializers(self):
        return self._initializers

    def process(self, *args, **kwargs):
        raise NotImplementedError("process is abstract")

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def to_dictionary(self, session):
        raise NotImplementedError("to_dictionary is abstract")

    @classmethod
    def from_dictionary(cls, data_dict):
        raise NotImplementedError("from_dictionary is abstract")

    @staticmethod
    def _write_dict_to_hdf_group(data_dict, group):
        for name, data in data_dict.items():
            if isinstance(data, dict):
                Unit._write_dict_to_hdf_group(data, group.create_group(name))
            else:
                group.create_dataset(name, data=data)

    @staticmethod
    def _load_hdf_group_to_dict(group):
        data_dict = {}
        for name, subgroup in group.items():
            if isinstance(subgroup, h5py.Dataset):
                data_dict[name] = subgroup.value
            else:
                data_dict[name] = Unit._load_hdf_group_to_dict(subgroup)
        return data_dict

    def save(self, session, file):
        data_dict = self.to_dictionary(session)
        with h5py.File(file, "w") as f:
            Unit._write_dict_to_hdf_group(data_dict, f)

    @classmethod
    def from_file(cls, file):
        with h5py.File(file, "r") as f:
            unit = cls.from_dictionary(Unit._load_hdf_group_to_dict(f))
        return unit

pass


class StackedUnits(Unit):

    def __init__(self, units=None):
        super().__init__()
        self._units = units if units is not None else []
        for u in self._units:
            self.register_subunit(u)

    @property
    def units(self):
        return self._units

    def add(self, unit):
        self._units.append(unit)
        self.register_subunit(unit)

    def process(self, inputs):
        outputs = inputs
        for unit in self.units:
            outputs = unit.process(outputs)
        return outputs

    def to_dictionary(self, session):
        return {
            "stacked_unit_%i" % (i+1): {
                "unit": unit.to_dictionary(session),
                "type": type(unit).__name__
            } for i, unit in enumerate(self.units)
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        stacked_units = []
        with tf.variable_scope(scope, default_name="stacked_units"):
            for i in range(len(data_dict)):
                unit_dict = data_dict["stacked_unit_%i"%(i+1)]
                unit_data_dict = unit_dict["unit"]
                unit_cls_string = unit_dict["type"]
                unit_cls = Unit.class_by_name(unit_cls_string)
                unit = unit_cls.from_dictionary(unit_data_dict)
                stacked_units.append(unit)
        return cls(stacked_units)

pass


class RecurrentUnit(Unit):

    def __init__(self, cell):
        super().__init__()
        self._cell = cell
        self.register_subunit(cell)

    @property
    def cell(self):
        return self._cell

    def process(self, inputs, initial_state=None, state=None, include_state=False, scope=None):
        with tf.variable_scope(scope, default_name="rnn_output"):
            outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=initial_state, scope=scope)
        return (outputs if not include_state else (outputs, state))

    def to_dictionary(self, session):
        return {
            "cell": self.cell.to_dictionary(session),
            "type": type(self.cell).__name__
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope):
        cell_cls_string = data_dict["type"]
        cell_cls = Unit.class_by_name(cell_cls_string)
        cell_data_dict = data_dict["cell"]
        with tf.variable_scope(scope, default_name="recurrent_unit"):
            cell =  cell_cls.from_dictionary(cell_data_dict)
        return cls(cell)

pass


class StatefulUnit(RecurrentUnit):

    def __init__(self, cell, state_tuple=None):
        super().__init__(cell)
        self._states = state_tuple if state_tuple is not None else ()
        for state in [s for s in self._states if s is not None]:
            self.register_variable(state)

    @property
    def states(self):
        return self._states

    def reset_state(self, scope=None):
        with tf.variable_scope(scope, default_name="statefulunit_reset"):
            reset =[tf.assign(s, tf.zeros(s.get_shape())) for s in self.states if s is not None]
        return reset

    def process(self, inputs, save_state=True, include_state=False, scope=None):
        with tf.variable_scope(scope, default_name="statefulunit_output") as scope:
            initial_state = tuple([s if s is not None else tf.zeros([tf.shape(inputs)[0], d], tf.float32)
                                   for s, d in zip(list(self.states), list(self.cell.state_size))])
            outputs, state = super().process(inputs, initial_state=initial_state, include_state=True, scope=scope)
            if save_state:
                with tf.control_dependencies([tf.assign(s, ns) for s, ns in zip(self.states, state) if s is not None]):
                    outputs = tf.identity(outputs)
        return (outputs if not include_state else (outputs, state))

pass

