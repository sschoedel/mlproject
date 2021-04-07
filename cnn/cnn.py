import tensorflow as tf
import tensorflow.keras as keras
import os

# load the images
img_height = 120
img_width = 120
channels = 3
batch_size = 16

image_shape = (img_height, img_width, channels)
num_classes = 11

train_dir = os.path.join(os.getcwd(), 'images', 'train')
test_dir = os.path.join(os.getcwd(), 'images', 'test')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# construct the model
# layers can be added incrementally to Sequential object
model = models.Sequential()
model.add(layers.Conv2D(32, 3, activation='relu', input_shape=image_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, 3, activation='relu'))

# finish the model with dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# final layer should have nodes == number of classes
model.add(layers.Dense(num_classes))

# compile the model
# use SparseCategoricalCrossentropy when doing more than binary class classification
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

EPOCHS = 1000
history = model.fit(train_ds, epochs=EPOCHS)

# save the model
model.save(os.path.join(os.getcwd()), 'trained_model')
