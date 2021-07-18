# Import Required libraries
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import CSVLogger

# Setting up the data loaders
train_dir = '10_food_classes_10_percent/train'
test_dir = '10_food_classes_10_percent/test'

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print('Loading training data...')
train_data_10_percent = keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                         label_mode='categorical',
                                                                         image_size=IMAGE_SIZE,
                                                                         batch_size=BATCH_SIZE)
print('Loading test data...')
test_data = keras.preprocessing.image_dataset_from_directory(test_dir,
                                                             label_mode='categorical',
                                                             image_size=IMAGE_SIZE,
                                                             batch_size=BATCH_SIZE)

# Create data augmentation layer
data_augmentation = keras.Sequential([
                                      preprocessing.RandomFlip('horizontal'),
                                      preprocessing.RandomZoom(0.2),
                                      preprocessing.RandomRotation(0.2),
                                      preprocessing.RandomWidth(0.3),
                                      preprocessing.RandomHeight(0.3)], name='data_augmentation_layer')

# Create the base model from keras.application
base_model = keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# Create the inputs and outputs
input_shape = IMAGE_SIZE + (3,)

inputs = keras.layers.Input(shape=input_shape, name='input_layer')
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
outputs = keras.layers.Dense(10, activation='softmax', name='output_layer')(x)

model_0 = keras.Model(inputs, outputs)

# Compile the model
model_0.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# Create Model Checkpoint
checkpoint_path = 'checkpoints/model_0_weights.ckpt'

checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      verbose=1,
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      save_freq='epoch')

# Create CSV LOGGER callback for history capture
csvlogger = CSVLogger('training.log', separator=',', append=True)

# Fit Model 0 with the Model checkpoint callback
initial_epochs = 10
history_0 = model_0.fit(train_data_10_percent,
                        epochs=initial_epochs,
                        steps_per_epoch=len(train_data_10_percent),
                        validation_data=test_data,
                        validation_steps=int(0.25 * len(test_data)),
                        callbacks=[checkpoint_callback, csvlogger])

initial_result = model_0.evaluate(test_data)
print(initial_result)
model_0.save('initial_model_without_tuning.h5')

# Opening the last 20 layers of the base model for training and recompiling the model with revised 10x decreased
# learning rate

# Load Saved Model
# model_0 = keras.models.load_model('initial_model_without_tuning.h5')

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model_0.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

print(f'Trainable Parameters : {len(model_0.trainable_variables)}')

for layer_number, layer in enumerate(model_0.layers):
    print(layer_number, layer.name, layer.trainable)

# refitting the model to start learning from where it had left earlier and fine-tune for additional 10 epochs
second_epochs = initial_epochs + 10

history_1 = model_0.fit(train_data_10_percent,
                        epochs=second_epochs,
                        steps_per_epoch=len(train_data_10_percent),
                        validation_data=test_data,
                        validation_steps=len(test_data),
                        initial_epoch=history_0.epoch[-1],
                        callbacks=[checkpoint_callback, csvlogger])

# Save the fine-tuned model
model_0.save('fine_tuned_model_with_20epochs.h5')

print('Before opening the last 30 layers \n')
for layer_number, layer in enumerate(base_model.layers):
    print( layer_number, layer.name, layer.trainable)
# Open another 10 more layers from the base model and retrain
for layer in base_model.layers[-30:]:
    layer.trainable = True

print('After opening the last 30 layers \n')
for layer_number, layer in enumerate(base_model.layers):
    print(layer_number, layer.name, layer.trainable)

# Compile the model with decreased learning rate as new layers were opened
model_0.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

# Refit the model
final_epochs = second_epochs + 10
history_2 = model_0.fit(train_data_10_percent,
                        epochs=final_epochs,
                        steps_per_epoch=len(train_data_10_percent),
                        validation_data=test_data,
                        validation_steps=int(0.25 * len(test_data)),
                        initial_epoch=history_1.epoch[-1],
                        callbacks=[checkpoint_callback, csvlogger])

model_0.save('final_fine_tuned_model_with_30epochs.h5')


