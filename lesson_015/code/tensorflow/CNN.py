# Import packages system 
import os

# Import charts
import matplotlib.pyplot as plt 

# Import packages to TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD


# Prepare set data
# To Train
#train_datagen = ImageDataGenerator(
#    rescale = 1.0/255.0
#)

train_datagen = ImageDataGenerator(rescale= 1.0/255.0,
                                    rotation_range =40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                    )

#To Test
test_datagen = ImageDataGenerator(
    rescale=1.0/255.0
)

# Flow training images in batches of 20 using train generator
train_generator = train_datagen.flow_from_directory('./data/train',
                                                    classes=['autos','ciudad'],
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150)
                                                    )

#Flow test datagen generator
validation_generator = test_datagen.flow_from_directory('./data/test',
                                                        classes=['autos','ciudad'],
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150,150)
                                                        )


#Model
#------------------------------------------------------------------------------------------
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  # The second convolution
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # The third convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # The fourth convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  # # The fifth convolution
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Dropout(0.2),
  # Flatten the results to feed into a DNN
  tf.keras.layers.Flatten(),
  # 512 neuron hidden layer
  tf.keras.layers.Dense(512, activation='relu'),
  # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('autos') and 1 for the other ('ciudad')
  tf.keras.layers.Dense(1, activation='sigmoid')
])
#--------------------------------------------------------------------------------------------

# Compiler Model
#-----------------------------------------------------
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics='accuracy')

#-----------------------------------------------------

# Train model
#------------------------------------------------------
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=15,
                    validation_steps=50,
                    verbose=1)
#------------------------------------------------------

# Save Setup Model

model_json = model.to_json()

with open("model.json","w") as json_file:
    json_file.write(model_json)

# Save Weights Model
model.save_weights("model.h5")

print("Model Save in this PC")

#Show data Train
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epo = list(range(len(acc)))

plt.plot(epo,acc,label='Training acc')
plt.plot(epo,val_acc,label="Validation acc")

plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')

plt.legend()
plt.show()