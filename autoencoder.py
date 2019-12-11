{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Keras AutoEncoder \n",
    "\n",
    "Autoencoder in keras to test GPU instances on AWS/GCP"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import keras\n",
    "from matplotlib import pyplot as plt\n",
    "import numpy as np\n",
    "import gzip\n",
    "%matplotlib inline\n",
    "from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D\n",
    "from keras.models import Model\n",
    "from keras.optimizers import RMSprop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_data(filename, num_images):\n",
    "    with gzip.open(filename) as bytestream:\n",
    "        bytestream.read(16)\n",
    "        buf = bytestream.read(28 * 28 * num_images)\n",
    "        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)\n",
    "        data = data.reshape(num_images, 28,28)\n",
    "        return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = extract_data('train-images-idx3-ubyte.gz', 60000)\n",
    "test_data = extract_data('t10k-images-idx3-ubyte.gz', 10000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_labels(filename, num_images):\n",
    "    with gzip.open(filename) as bytestream:\n",
    "        bytestream.read(8)\n",
    "        buf = bytestream.read(1 * num_images)\n",
    "        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)\n",
    "        return labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_labels = extract_labels('train-labels-idx1-ubyte.gz',60000)\n",
    "test_labels = extract_labels('t10k-labels-idx1-ubyte.gz',10000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Shapes of training set\n",
    "print(\"Training set (images) shape: {shape}\".format(shape=train_data.shape))\n",
    "\n",
    "# Shapes of test set\n",
    "print(\"Test set (images) shape: {shape}\".format(shape=test_data.shape))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create dictionary of target classes\n",
    "label_dict = {\n",
    " 0: 'A',\n",
    " 1: 'B',\n",
    " 2: 'C',\n",
    " 3: 'D',\n",
    " 4: 'E',\n",
    " 5: 'F',\n",
    " 6: 'G',\n",
    " 7: 'H',\n",
    " 8: 'I',\n",
    " 9: 'J',\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=[5,5])\n",
    "\n",
    "# Display the first image in training data\n",
    "plt.subplot(121)\n",
    "curr_img = np.reshape(train_data[0], (28,28))\n",
    "curr_lbl = train_labels[0]\n",
    "plt.imshow(curr_img, cmap='gray')\n",
    "plt.title(\"(Label: \" + str(label_dict[curr_lbl]) + \")\")\n",
    "\n",
    "# Display the first image in testing data\n",
    "plt.subplot(122)\n",
    "curr_img = np.reshape(test_data[0], (28,28))\n",
    "curr_lbl = test_labels[0]\n",
    "plt.imshow(curr_img, cmap='gray')\n",
    "plt.title(\"(Label: \" + str(label_dict[curr_lbl]) + \")\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = train_data.reshape(-1, 28,28, 1)\n",
    "test_data = test_data.reshape(-1, 28,28, 1)\n",
    "train_data.shape, test_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data.dtype, test_data.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.max(train_data), np.max(test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = train_data / np.max(train_data)\n",
    "test_data = test_data / np.max(test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.max(train_data), np.max(test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,\n",
    "                                                             train_data, \n",
    "                                                             test_size=0.2, \n",
    "                                                             random_state=13)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 128\n",
    "epochs = 10\n",
    "inChannel = 1\n",
    "x, y = 28, 28\n",
    "input_img = Input(shape = (x, y, inChannel))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def autoencoder(input_img):\n",
    "    #encoder\n",
    "    #input = 28 x 28 x 1 (wide and thin)\n",
    "    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32\n",
    "    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32\n",
    "    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64\n",
    "    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64\n",
    "    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)\n",
    "\n",
    "    #decoder\n",
    "    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128\n",
    "    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128\n",
    "    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64\n",
    "    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64\n",
    "    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1\n",
    "    return decoded"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "autoencoder = Model(input_img, autoencoder(input_img))\n",
    "autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "autoencoder.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "autoencoder_train = autoencoder.fit(train_X, \n",
    "                                    train_ground, \n",
    "                                    batch_size=batch_size,\n",
    "                                    epochs=epochs,\n",
    "                                    verbose=1,\n",
    "                                    validation_data=(valid_X, valid_ground))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "loss = autoencoder_train.history['loss']\n",
    "val_loss = autoencoder_train.history['val_loss']\n",
    "epochs = range(epochs)\n",
    "plt.figure()\n",
    "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
    "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
    "plt.title('Training and validation loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred = autoencoder.predict(test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(20, 4))\n",
    "print(\"Test Images\")\n",
    "for i in range(10):\n",
    "    plt.subplot(2, 10, i+1)\n",
    "    plt.imshow(test_data[i, ..., 0], cmap='gray')\n",
    "    curr_lbl = test_labels[i]\n",
    "    plt.title(\"(Label: \" + str(label_dict[curr_lbl]) + \")\")\n",
    "plt.show()    \n",
    "plt.figure(figsize=(20, 4))\n",
    "print(\"Reconstruction of Test Images\")\n",
    "for i in range(10):\n",
    "    plt.subplot(2, 10, i+1)\n",
    "    plt.imshow(pred[i, ..., 0], cmap='gray')  \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "noise_factor = 0.5\n",
    "x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_X.shape)\n",
    "x_valid_noisy = valid_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=valid_X.shape)\n",
    "x_test_noisy = test_data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_data.shape)\n",
    "x_train_noisy = np.clip(x_train_noisy, 0., 1.)\n",
    "x_valid_noisy = np.clip(x_valid_noisy, 0., 1.)\n",
    "x_test_noisy = np.clip(x_test_noisy, 0., 1.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=[5,5])\n",
    "\n",
    "# Display the first image in training data\n",
    "plt.subplot(121)\n",
    "curr_img = np.reshape(x_train_noisy[1], (28,28))\n",
    "plt.imshow(curr_img, cmap='gray')\n",
    "\n",
    "# Display the first image in testing data\n",
    "plt.subplot(122)\n",
    "curr_img = np.reshape(x_test_noisy[1], (28,28))\n",
    "plt.imshow(curr_img, cmap='gray')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 128\n",
    "epochs = 20\n",
    "inChannel = 1\n",
    "x, y = 28, 28\n",
    "input_img = Input(shape = (x, y, inChannel))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def autoencoder2(input_img):\n",
    "    #encoder\n",
    "    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)\n",
    "    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)\n",
    "    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)\n",
    "    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)\n",
    "    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)\n",
    "\n",
    "    #decoder\n",
    "    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)\n",
    "    up1 = UpSampling2D((2,2))(conv4)\n",
    "    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)\n",
    "    up2 = UpSampling2D((2,2))(conv5)\n",
    "    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)\n",
    "    return decoded"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "autoencoder = Model(input_img, autoencoder2(input_img))\n",
    "autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "autoencoder_train = autoencoder.fit(x_train_noisy, train_X, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid_noisy, valid_X))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "loss = autoencoder_train.history['loss']\n",
    "val_loss = autoencoder_train.history['val_loss']\n",
    "epochs = range(epochs)\n",
    "plt.figure()\n",
    "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
    "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
    "plt.title('Training and validation loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred = autoencoder.predict(x_test_noisy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(20, 4))\n",
    "print(\"Test Images\")\n",
    "for i in range(10,20,1):\n",
    "    plt.subplot(2, 10, i+1)\n",
    "    plt.imshow(test_data[i, ..., 0], cmap='gray')\n",
    "    curr_lbl = test_labels[i]\n",
    "    plt.title(\"(Label: \" + str(label_dict[curr_lbl]) + \")\")\n",
    "plt.show()    \n",
    "plt.figure(figsize=(20, 4))\n",
    "print(\"Test Images with Noise\")\n",
    "for i in range(10,20,1):\n",
    "    plt.subplot(2, 10, i+1)\n",
    "    plt.imshow(x_test_noisy[i, ..., 0], cmap='gray')\n",
    "plt.show()    \n",
    "\n",
    "plt.figure(figsize=(20, 4))\n",
    "print(\"Reconstruction of Noisy Test Images\")\n",
    "for i in range(10,20,1):\n",
    "    plt.subplot(2, 10, i+1)\n",
    "    plt.imshow(pred[i, ..., 0], cmap='gray')  \n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
