# Code from: https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
# See also: https://blog.keras.io/building-autoencoders-in-keras.html
#           https://www.tensorflow.org/tutorials/generative/autoencoder

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Input, Flatten, Dense, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import os.path
import math

# Define size of encoded layer
encoding_size = 256

# Functions


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)
        return data


def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels


def encode(input_img):
    # input = 28 x 28 x 1
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    flat = Flatten()(conv3)
    dense = Dense(encoding_size, activation='relu')(flat)
    return dense


def decode(input_code):
    # input = encoding_size x 1 x 1
    dense = Dense(3136, activation='relu')(input_code)
    reshape = Reshape((7, 7, 64))(dense)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(reshape)
    up1 = UpSampling2D((2, 2))(conv1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(conv2)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
    return decoded


# Create dictionary of target classes
label_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
}

# Import data
train_data = extract_data('notMNIST/train-images-idx3-ubyte.gz', 60000)
test_data = extract_data('notMNIST/t10k-images-idx3-ubyte.gz', 10000)
train_labels = extract_labels('notMNIST/train-labels-idx1-ubyte.gz', 60000)
test_labels = extract_labels('notMNIST/t10k-labels-idx1-ubyte.gz', 10000)

# Prepare training and validation data
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

train_X, valid_X, train_ground, valid_ground = train_test_split(
    train_data, train_data, test_size=0.2)

# Training configuration
batch_size = 128
epochs = 50
inChannel = 1
x, y = 28, 28
input_img = Input(shape=(x, y, inChannel))
input_code = Input(shape=(encoding_size))

# Setup checkpoints
checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create models
encoded = encode(input_img)
encoder = Model(input_img, encoded)
decoder = Model(input_code, decode(input_code))
autoencoder = Model(input_img, decode(encoded))

# Compile models
autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
print(autoencoder.summary())

# Load previous checkpoint if it exists
if os.path.exists(checkpoint_dir):
    autoencoder.load_weights(checkpoint_path)
    print("Successfully loaded training checkpoint")

# Train the model
print("Epochs to train:")
epochs = int(input())
if (epochs > 0):
    autoencoder_train = autoencoder.fit(train_X,
                                        train_ground,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(valid_X,
                                                         valid_ground),
                                        callbacks=[cp_callback])

    # Plot loss
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# Predict
encoded_imgs = encoder.predict(test_data)
decoded_imgs = decoder.predict(encoded_imgs)

# Display results
n = 10
plt.figure(figsize=(18, 6))
for i in range(n):
    # Display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display encoded format
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(16, math.ceil(encoding_size / 16)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + n * 2)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
