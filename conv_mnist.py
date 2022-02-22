
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models

# -------------------------------------------------


''' PROGRAM OUTPUT
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 6)         156
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 6)         0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 14, 16)        2416
_________________________________________________________________
flatten_1 (Flatten)          (None, 3136)              0
_________________________________________________________________
dense_1 (Dense)              (None, 120)               376440
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1210
=================================================================
Total params: 380,222
Trainable params: 380,222
Non-trainable params: 0
_________________________________________________________________
optimizer='adam', loss='categorical_crossentropy'

Train on 48000 samples, validate on 12000 samples
Test on 10000 samples

48000/48000 [==============================] - 19s 400us/step - loss: 0.1608 - accuracy: 0.9519 - val_loss: 0.0611 - val_accuracy: 0.9810
Epoch 2/10
48000/48000 [==============================] - 22s 450us/step - loss: 0.0514 - accuracy: 0.9837 - val_loss: 0.0656 - val_accuracy: 0.9791
Epoch 3/10
48000/48000 [==============================] - 21s 431us/step - loss: 0.0351 - accuracy: 0.9883 - val_loss: 0.0412 - val_accuracy: 0.9872
Epoch 4/10
48000/48000 [==============================] - 20s 414us/step - loss: 0.0249 - accuracy: 0.9922 - val_loss: 0.0410 - val_accuracy: 0.9881
Epoch 5/10
48000/48000 [==============================] - 20s 413us/step - loss: 0.0185 - accuracy: 0.9944 - val_loss: 0.0436 - val_accuracy: 0.9884
Epoch 6/10
48000/48000 [==============================] - 21s 433us/step - loss: 0.0137 - accuracy: 0.9952 - val_loss: 0.0474 - val_accuracy: 0.9885
Epoch 7/10
48000/48000 [==============================] - 20s 413us/step - loss: 0.0122 - accuracy: 0.9957 - val_loss: 0.0661 - val_accuracy: 0.9834
Epoch 8/10
48000/48000 [==============================] - 20s 415us/step - loss: 0.0108 - accuracy: 0.9965 - val_loss: 0.0443 - val_accuracy: 0.9891
Epoch 9/10
48000/48000 [==============================] - 21s 428us/step - loss: 0.0074 - accuracy: 0.9975 - val_loss: 0.0580 - val_accuracy: 0.9876
Epoch 10/10
48000/48000 [==============================] - 19s 397us/step - loss: 0.0081 - accuracy: 0.9972 - val_loss: 0.0586 - val_accuracy: 0.9868
10000/10000 [==============================] - 1s 147us/step
Test accuracy: 0.9879000186920166
'''

# Function to display plot
def display_metrics(history):

	# Set graph limits
	plt.ylim(top=1)
	plt.ylim(bottom=0)

	# Display n_iterations vs accuracy, validation-accuracy
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()

	# Display n_iterations vs loss, validation-loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()


# Function to create model for CNN
def get_model(x, y, out_classes):

	# Instantiate the model
	model = models.Sequential()

	# Input layer (size = 28x28)
	# Convolution layer (input channel = 6, size = 5x5, padding = 2, activation = 'relu')			## padding = (f-1)/2, where f = 5
	model.add(layers.Conv2D(6, kernel_size=(5, 5), padding = 'same', activation='relu', input_shape=(x, y, 1)))

	# Pooling layer (Pooling Method = Max-pooling, stride = 2)
	model.add(layers.MaxPooling2D(strides=(2, 2)))

	# Convolution layer (input channel = 16, size = 5x5, padding = 1, activation = 'relu')
	model.add(layers.Conv2D(16, kernel_size=(5, 5), padding = 'same', activation='relu'))

	# Flatten the 2D layers
	model.add(layers.Flatten())

	# Dense layer (size = 120, activation = 'relu')
	model.add(layers.Dense(120, activation='relu'))

	# Dense layer (size = 10) [output layer, number of digits = 10]
	model.add(layers.Dense(out_classes, activation='softmax'))

	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Print model summary
	print(model.summary())

	# Return the created model
	return model


# -------------------------------------------------

# Create model
model = get_model(28, 28, 10)

# Get the MNIST Train and Test datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape dataset for model from 784 to 28x28
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

# Convert to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the model on train data
history = model.fit(train_images, train_labels, epochs=10, validation_split = 0.2, batch_size=32, shuffle=True)

# Test the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print ('Test accuracy:', test_acc)

# Display training metrics
display_metrics(history)
