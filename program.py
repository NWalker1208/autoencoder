# Code from: https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import os.path


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


def autoencoder(input_img):
    # encoder
    # input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu',
                   padding='same')(input_img)  # 28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu',
                   padding='same')(pool1)  # 14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(
        pool2)  # 7 x 7 x 128 (small and thick)

    # decoder
    conv4 = Conv2D(128, (3, 3), activation='relu',
                   padding='same')(conv3)  # 7 x 7 x 128
    up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), activation='relu',
                   padding='same')(up1)  # 14 x 14 x 64
    up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid',
                     padding='same')(up2)  # 28 x 28 x 1
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

train_X, valid_X, train_ground, valid_ground = train_test_split(train_data,
                                                                train_data,
                                                                test_size=0.2,
                                                                random_state=13)

# Training configuration
batch_size = 128
epochs = 50
inChannel = 1
x, y = 28, 28
input_img = Input(shape=(x, y, inChannel))

# Setup checkpoints
checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create model
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

# Load previous checkpoint if it exists
do_training = False
if os.path.exists(checkpoint_dir):
    autoencoder.load_weights(checkpoint_path)
    print("Successfully loaded previous training. Skip training? (Y/N) ")
    do_training = (input() == "N")

# Train the model
if do_training:
    autoencoder_train = autoencoder.fit(train_X,
                                        train_ground,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=1,
                                        validation_data=(
                                            valid_X, valid_ground),
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
pred = autoencoder.predict(test_data)
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(test_data[i, ..., 0], cmap='gray')
    curr_lbl = test_labels[i]
    plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
plt.show()
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')
plt.show()
