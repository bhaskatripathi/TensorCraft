from .unit import *
import tensorflow as tf
from math import sqrt
import numpy as np


class StringHashProducer(Unit):

    def __init__(self, n_buckets):
        super().__init__()
        self._n_buckets = n_buckets

    @property
    def n_buckets(self):
        return self._n_buckets

    def process(self, inputs, scope=None):
        with tf.variable_scope(scope, default_name="string_hash"):
            hashed = tf.string_to_hash_bucket_fast(inputs, self.n_buckets)
        return hashed

    def to_dictionary(self, session=None):
        return {"n_buckets": self.n_buckets}

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        return cls(data_dict["n_buckets"])


pass


class IndexMap(Unit):

    def __init__(self, mapping, default_value=-1):
        super().__init__()
        self._mapping = mapping
        self.register_variable(mapping)
        self._default_value = default_value
        self._table = tf.contrib.lookup.index_table_from_tensor(
            mapping.initialized_value(),
            default_value=default_value,
            dtype=tf.int64
        )
        self.register_initializer(self.table.init)

    @classmethod
    def from_arrays(cls, mapping_array, default_value=-1, scope=None):
        with tf.variable_scope(scope, default_name="indexmap"):
            mapping = tf.get_variable(
                "mapping",
                shape=mapping_array.shape,
                dtype=tf.int64,
                initializer=tf.constant_initializer(mapping_array),
                trainable=False
            )
        return cls(mapping, default_value)

    @property
    def mapping(self):
        return self._mapping

    @property
    def table(self):
        return self._table

    @property
    def default_value(self):
        return self._default_value

    def process(self, inputs, scope=None):
        with tf.variable_scope(scope, default_name="indexmap_output"):
            outputs = self.table.lookup(inputs)
        return outputs

    def to_dictionary(self, session):
        return {
            "mapping": session.run(self.mapping),
            "default_value": self.default_value
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        mapping_array = data_dict["mapping"]
        return cls.from_arrays(mapping_array, data_dict["default_value"], scope)


pass


class EmbeddingLayer(Unit):

    def __init__(self, embeddings, trainable=False):
        super().__init__()
        self._embeddings = embeddings
        self._trainable = trainable
        if trainable:
            self.register_parameter(embeddings)
        else:
            self.register_variable(embeddings)
        self._output_dim = tf.shape(self._embeddings)[-1]

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def trainable(self):
        return self._trainable

    @classmethod
    def from_arrays(cls, embeddings_array, trainable=False, scope=None):
        with tf.variable_scope(scope, default_name="embedding_layer"):
            embeddings = tf.get_variable(
                "embeddings",
                shape=embeddings_array.shape,
                dtype=embeddings_array.dtype,
                initializer=tf.constant_initializer(embeddings_array),
                trainable=trainable
            )
        return cls(embeddings, trainable)

    @classmethod
    def from_description(cls, number, dim, trainable=True, scope=None):
        with tf.variable_scope(scope, default_name="embedding_layer"):
            embeddings = tf.get_variable(
                "embeddings",
                shape=[number, dim],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0.0, 1 / sqrt(dim)),
                trainable=trainable
            )
        return cls(embeddings, dim)

    def process(self, inputs, scope=None):
        with tf.variable_scope(scope, default_name="embedding_layer_outputs"):
            batch = len(inputs.shape) > 1
            if batch:
                outputs_shape = inputs.get_shape().as_list() + [self.output_dim.value]
                outputs_shape = [s if s is not None else -1 for s in outputs_shape]
                inputs = tf.reshape(inputs, [-1])
            adj_indeces = tf.map_fn(
                lambda i: tf.cond(
                    pred=tf.equal(i, -1),
                    true_fn=lambda: tf.cast(tf.shape(self.embeddings)[0], tf.int64),
                    false_fn=lambda: i
                ),
                elems=inputs
            )
            unk_embedding = tf.zeros([1, tf.shape(self.embeddings)[1]], dtype=tf.float32)
            adj_embeddings = tf.concat([self.embeddings, unk_embedding], axis=0)
            outputs = tf.gather(adj_embeddings, adj_indeces, axis=0)
            if batch:
                outputs = tf.reshape(outputs, outputs_shape)
        return outputs

    def to_dictionary(self, session):
        return {"embeddings": session.run(self.embeddings)}

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        return cls.from_arrays(data_dict["embeddings"])


pass


class TextEmbeddingLayer(Unit):

    def __init__(self, string_hash_producer, index_map, embedding_layer):
        super().__init__()
        self._string_hash_producer = string_hash_producer
        self._index_map = index_map
        self._embedding_layer = embedding_layer
        self.register_subunit(string_hash_producer)
        self.register_subunit(index_map)
        self.register_subunit(embedding_layer)

    @property
    def string_hash_producer(self):
        return self._string_hash_producer

    @property
    def index_map(self):
        return self._index_map

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @classmethod
    def from_embedding_layer(cls, vocabulary, embedding_layer, scope=None):
        with tf.variable_scope(scope, default_name="text_embedding_layer"):
            string_hash_producer = StringHashProducer(2 ** 63 - 1)
            mappings = tf.get_variable(
                "mappings",
                initializer=string_hash_producer.process(vocabulary),
                validate_shape=False,
                trainable=False
            )
            index_map = IndexMap(mappings, default_value=-1)
        return cls(string_hash_producer, index_map, embedding_layer)

    @classmethod
    def from_embeddings(cls, vocabulary, embeddings, trainable=False, scope=None):
        embedding_layer = EmbeddingLayer(embeddings, trainable)
        return cls.from_embedding_layer(vocabulary, embedding_layer, scope)

    @classmethod
    def from_description(cls, vocabulary, embedding_dim, trainable=True, scope=None):
        with tf.variable_scope(scope, default_name="text_embedding_layer") as scope:
            emb_layer = EmbeddingLayer.from_description(vocabulary.shape[0], embedding_dim, trainable)
        return cls.from_embedding_layer(vocabulary, emb_layer, scope)

    def process(self, inputs, scope=None):
        with tf.variable_scope(scope, default_name="text_embedding_layer_output"):
            batch = len(inputs.shape) > 1
            if batch:
                #outputs_shape = inputs.get_shape().as_list() + [self.embedding_layer.output_dim.value]
                #outputs_shape = [s if s is not None else -1 for s in outputs_shape]
                outputs_shape = [*tf.unstack(tf.shape(inputs)), tf.shape(self.embedding_layer.embeddings)[-1]]
                inputs = tf.reshape(inputs, [-1])
            hashes = self.string_hash_producer.process(inputs)
            indeces = self.index_map.process(hashes)
            outputs = self.embedding_layer.process(indeces)
            if batch:
                outputs = tf.reshape(outputs, outputs_shape)
                outputs.set_shape(outputs.get_shape().as_list()[:-1]+[self.embedding_layer.embeddings.shape[-1]])
        return outputs

    def to_dictionary(self, session):
        return {
            "string_hash_producer": self.string_hash_producer.to_dictionary(session),
            "index_map": self.index_map.to_dictionary(session),
            "embedding_layer": self.embedding_layer.to_dictionary(session)
        }

    @classmethod
    def from_dictionary(cls, data_dict, scope=None):
        with tf.variable_scope(scope, default_name="text_embedding_layer"):
            string_hasher = StringHashProducer.from_dictionary(data_dict["string_hash_producer"])
            index_map = IndexMap.from_dictionary(data_dict["index_map"])
            embedding_layer = EmbeddingLayer.from_dictionary(data_dict["embedding_layer"])
        return cls(string_hasher, index_map, embedding_layer)

pass