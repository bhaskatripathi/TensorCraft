# TensorCraft
TensorCraft is a Python library tailored for building, training, and deploying neural networks using TensorFlow. It provides high-level constructs specifically designed to enhance the customization and flexibility of model development. TensorCraft is engineered to cater to both general and specialized needs, offering advanced control over TensorFlow operations and configurations that go beyond the typical use cases addressed by Keras.

## Core Features and Philosophies

- **Advanced Customization**: Unlike Keras, which offers broad solutions applicable to a wide range of applications, TensorCraft allows deeper customization of neural network components. This is particularly useful for users who need to fine-tune their models to specific requirements or who are engaged in developing novel machine learning algorithms.

- **Optimized Constructs for Specialized Tasks**: TensorCraft includes unique constructs and functionalities optimized for particular types of data or tasks, such as complex sequence models or specialized embedding layers, providing tools that are particularly adept at handling advanced neural network challenges.

- **Flexible and Extensible Design**: Designed with extensibility in mind, TensorCraft facilitates easy modifications and extensions of its core components. This feature is essential for researchers and developers who require the ability to experiment with cutting-edge neural network architectures or to integrate new TensorFlow features as they become available.

- **Focused Community and Agile Development**: While Keras benefits from extensive documentation and a broad user base as part of the TensorFlow ecosystem, TensorCraft offers a more concentrated community focus. This leads to rapid iterations and the ability to quickly adapt to the latest research findings or industry demands, ensuring that users have access to the most effective and innovative tools.

## Features currently available

- **Modular Design**: Easily extendable to incorporate custom layers and functionalities.
- **User-friendly API**: Simplifies typical machine learning workflows, making development more intuitive.
- **Support for Advanced Architectures**: Includes tools for creating advanced neural network structures such as RNNs, LSTMs, and embedding layers.

## Installation
TensorCraft is easy to install and integrate into your existing TensorFlow workflows. To install TensorCraft, clone the repository and install using pip. Open your terminal and execute the following commands:

    git clone https://github.com/your-github-username/TensorCraft.git
    cd TensorCraft
    pip install .

## Example Usage

Here's a quick example showing how to set up a text classification model using pre-trained GloVe embeddings.

### Import Required Modules

First, import the necessary modules from TensorCraft along with TensorFlow:

    from TensorCraft.zoo import GloveEmbeddingLayer
    from TensorCraft.dense import DenseLayer, DenseClassifier
    import tensorflow as tf

### Load GloVe Embeddings

Assuming you have a CSV file containing GloVe embeddings, you can load them like this:

    glove_layer = GloveEmbeddingLayer.from_csv('path_to_glove.csv', embedding_dim=50, trainable=False)

### Define the Model

Set up a classifier with the Glove embedding layer:

    classifier = DenseClassifier.from_description(input_dim=50, num_classes=2, layer_sizes=[100], act_funcs=[tf.nn.relu])
    inputs = tf.placeholder(tf.int32, shape=[None, 10])  # batch_size x sequence_length
    labels = tf.placeholder(tf.float32, shape=[None, 2])  # batch_size x num_classes
    logits = classifier.process(inputs)

### Training Configuration

Configure the training operations:

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

### Train the Model

Execute the training session:

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(100):
            batch_inputs, batch_labels = get_next_batch()  # Implement this function
            _, loss_val = sess.run([optimizer, loss], feed_dict={inputs: batch_inputs, labels: batch_labels})
            print("Step:", step, "Loss:", loss_val)

## Contributing

Contributions are welcome! Please submit your pull requests or issues through GitHub.

## License

TensorCraft is released under the MIT License. See the LICENSE file for more details.
