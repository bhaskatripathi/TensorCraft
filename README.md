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

Here's a quick example showing how to set up a sequential data forecasting model for time series prediction using LSTM cells from TensorCraft.

### Import Required Modules

First, import the necessary modules from TensorCraft along with TensorFlow:

```python
from TensorCraft.recurrent import LSTMCell, StatefulUnit
from TensorCraft.dense import DenseLayer
import tensorflow as tf
import numpy as np
```

### Define the Model

```python
input_dim = 1  # Assuming one feature per time step
hidden_units = 50  # Number of LSTM units
output_dim = 1  # Forecasting one step ahead

# Create LSTM cell
lstm_cell = LSTMCell.from_description(input_dim + hidden_units, hidden_units, tf.tanh)

# Create stateful unit to maintain state across time steps
lstm_network = StatefulUnit(lstm_cell)

# Define dense layer to map LSTM outputs to the forecasted values
output_layer = DenseLayer.from_description(hidden_units, output_dim, act=None)
```

### Load your data

```python
data = np.sin(np.linspace(0, 100, 1000))  # Example data: Sine wave
sequence_length = 10  # Use 10 time steps to predict the next step

# Function to create sequences from data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

x_train, y_train = create_sequences(data, sequence_length)
x_train = np.expand_dims(x_train, axis=2)  # Shape: [num_samples, sequence_length, 1]
y_train = np.expand_dims(y_train, axis=1)  # Shape: [num_samples, 1]
```

### Training Configuration

#Configure the training operations:
```python
inputs = tf.placeholder(tf.float32, [None, sequence_length, 1])
labels = tf.placeholder(tf.float32, [None, 1])
outputs = lstm_network.process(inputs)[-1]  # Only take the last output for prediction
predictions = output_layer.process(outputs)

loss = tf.reduce_mean(tf.square(predictions - labels))  # Mean squared error
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
```

### Train the Model
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        _, loss_val = sess.run([optimizer, loss], feed_dict={inputs: x_train, labels: y_train})
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss_val}')
```

## Contributing

Contributions are welcome! Please submit your pull requests or issues through GitHub. We aim to improve this library and make it an alternative to Keras.

## License

TensorCraft is released under the MIT License. See the LICENSE file for more details.
