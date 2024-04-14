from .feature import TextEmbeddingLayer
import tensorflow as tf


class GloveEmbeddingLayer(TextEmbeddingLayer):

    @classmethod
    def from_csv(cls, file_name, embedding_dim, trainable=False, scope=None):
        with tf.variable_scope(scope, default_name="glove_embedding_layer") as scope:
            csv_file = tf.read_file(file_name)
            csv_lines = tf.string_split([csv_file], delimiter="\n")
            csv_format = [tf.constant([], dtype=tf.string)] + [tf.constant([], dtype=tf.float32)] * embedding_dim
            csv_columns = tf.decode_csv(csv_lines.values, csv_format, field_delim=' ', use_quote_delim=False)
            glove_vocabulary = csv_columns[0]
            glove_embeddings = tf.get_variable(
                "glove_embeddings",
                initializer=tf.stack(csv_columns[1:], axis=1),
                validate_shape=False,
                trainable=trainable
            )
            glove_embeddings.set_shape([None, embedding_dim])
        return cls.from_embeddings(glove_vocabulary, glove_embeddings, trainable, scope)

pass