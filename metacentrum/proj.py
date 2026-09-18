import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input

# Load data for pairs creation
original_dir = os.path.join('data', 'Original')
deepfake_dir = os.path.join('data', 'Deepfake')

# List all files in the directories
original_files = [os.path.join(original_dir, f) for f in os.listdir(original_dir)]
deepfake_files = [os.path.join(deepfake_dir, f) for f in os.listdir(deepfake_dir)]

# Pairs creation
# Original vs Deepfake of Original (Label 1)
pairs = []
labels = []
for original_file in original_files:
    filename = os.path.basename(original_file)
    index = filename.split('_')[1].split('.')[0]

    # Construct paths for the corresponding deepfake images
    deepfake_file1 = os.path.join(deepfake_dir, f'Deep1_{index}.jpg')
    deepfake_file2 = os.path.join(deepfake_dir, f'Deep2_{index}.jpg')
    deepfake_file3 = os.path.join(deepfake_dir, f'Deep3_{index}.jpg')
    deepfake_file4 = os.path.join(deepfake_dir, f'Deep4_{index}.jpg')

    if deepfake_file3 in deepfake_files:
        pairs.append((original_file, deepfake_file3))
        labels.append(1)

    # Check if both deepfake images exist
    if deepfake_file1 in deepfake_files and deepfake_file2 in deepfake_files and deepfake_file4 in deepfake_files:
        pairs.append((original_file, deepfake_file1))
        labels.append(1)  # 1 for similar pairs (original and deepfake)

        pairs.append((original_file, deepfake_file2))
        labels.append(1)  # 1 for similar pairs (original and deepfake)

        pairs.append((original_file, deepfake_file4))
        labels.append(1)  # 1 for similar pairs (original and deepfake)

# Creation of pairs as label 0
# Original vs. Deepfake of Another Person (Label 0)

# The third deepfake algorithm was able to produce deepfakes only for LFW dataset. 
# Thats why it is separated
index = 700
for original_file in original_files:
    filename = os.path.basename(original_file)

    deepfake_file3 = os.path.join(deepfake_dir, f'Deep3_{index}.jpg')

    if deepfake_file3 in deepfake_files:
        pairs.append((original_file, deepfake_file3))
        labels.append(0)
    index -= 1

index = 1400
for original_file in original_files:
    filename = os.path.basename(original_file)

    # Construct paths for the corresponding deepfake images
    deepfake_file1 = os.path.join(deepfake_dir, f'Deep1_{index}.jpg')
    deepfake_file2 = os.path.join(deepfake_dir, f'Deep2_{index}.jpg')
    deepfake_file4 = os.path.join(deepfake_dir, f'Deep4_{index}.jpg')

    # Check if the deepfake image of another person exists
    if deepfake_file1 in deepfake_files and deepfake_file2 in deepfake_files and deepfake_file4 in deepfake_files:
        pairs.append((original_file, deepfake_file1))
        labels.append(0)  # 0 for dissimilar pairs (original and deepfake of another person)

        pairs.append((original_file, deepfake_file2))
        labels.append(0)  # 0 for dissimilar pairs (original and deepfake of another person)

        pairs.append((original_file, deepfake_file4))
        labels.append(0)  # 0 for dissimilar pairs (original and deepfake of another person)
    index -= 1

# Original vs. Original (Label 0)
for original_file in original_files:
    pairs.append((original_file, original_file))
    labels.append(0)  # 0 for dissimilar pairs (original vs. original)


# Procesing image with tf functions and transform to (0,1)
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Preparing pairs for convertion to numpy array
def load_and_preprocess_pairs(pairs):
    scaled_pairs = []
    for pair in pairs:
        img1 = preprocess_image(pair[0])
        img2 = preprocess_image(pair[1])
        scaled_pairs.append((img1, img2))
    return scaled_pairs

scaled_pairs = load_and_preprocess_pairs(pairs)
labels = np.array(labels)

# Shuffle and split the data
dataset = tf.data.Dataset.from_tensor_slices((scaled_pairs, labels))
dataset = dataset.shuffle(buffer_size=len(scaled_pairs))

train_size = int(len(scaled_pairs) * 0.7)
val_size = int(len(scaled_pairs) * 0.2)
test_size = len(scaled_pairs) - train_size - val_size

train_data = dataset.take(train_size)
remaining_data = dataset.skip(train_size)
val_data = remaining_data.take(val_size)
test_data = remaining_data.skip(val_size)

batch_size = 64 # For ResNet50 the value was 128 to improve the performance faster
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)
test_data = test_data.batch(batch_size)

from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications import VGG19 #ResNet50, VGG16
from tensorflow.keras import layers

# Definition of model
def custom_model(inputs):
    # vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3)) Model with ImageNet weights
    vgg_model = VGG19(include_top=False, input_shape=(128, 128, 3))

    for layer in vgg_model.layers:
        layer.trainable = False
    # inputs = Input(shape=(128,128,3))
    pair1 = input_pairs[:, 0, :, :, :]
    pair2 = input_pairs[:, 1, :, :, :]

    vgg_output_1 = vgg_model(pair1)
    flatten1 = Flatten()(vgg_output_1)
    dense1 = Dense(4096, activation='sigmoid')(flatten1)

    vgg_output_2 = vgg_model(pair2)
    flatten2 = Flatten()(vgg_output_2)
    dense2 = Dense(4096, activation='sigmoid')(flatten2)

    distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([dense1, dense2]) # TODO: mozno lepsie premenovat premenne
    return distance

from keras.optimizers import Adam

optimizer = Adam(learning_rate = 1e-4)

# Contrastive loss function definition
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    square_predict = tf.square(y_pred)

    loss = tf.reduce_mean((1 - y_true) * square_predict + y_true * tf.square(tf.maximum(0.0, margin - y_pred)))
    return loss

# Create siamese model architecture
input_pairs = tf.keras.layers.Input(shape=(2, 128, 128, 3))

distance = custom_model(input_pairs)

output = Dense(1, activation='sigmoid')(distance)

# Create siamese model
siamese_model = Model(inputs=input_pairs, outputs=output)

siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) #contrastive_loss

siamese_model.summary()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# For ResNet50 model the epochs were changed to 500
siamese_history = siamese_model.fit(train_data, validation_data=val_data, epochs=35, callbacks=[tensorboard_callback]) #35

test_loss, test_acc = siamese_model.evaluate(test_data)
print('Test accuracy:', test_acc)

# Saving of model
siamese_model.save(os.path.join('model','siamese_model.h5')) # .h5 for keras models

# This part bellow is only to evaluate performance of model. It was used only for ResNet50 model, 
# because we were not able to run it locally

from PIL import Image   # PIL library import

# This one for cycle is similar to others
# The first cylce is working with LFW dataset -> for index in range(1, 701)
# The second one is working with CelebA HQ dataset -> for index in range(701, 1400):
count = 0
for index in range(1, 701):
    # Loading corresponding pair
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Deepfake/Deep1_{index}.jpg')

    # Check if the images exist
    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    # Normalization
    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    # Connection to pair
    image_pair = np.stack([img1, img2], axis=0)

    # Prediction
    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    # Model evaluation
    evaluation = predictions[0][0]

    # If deepfake we increment count value
    if evaluation >= 0.5:
        count += 1

ratio1 = count / 700

count = 0
for index in range(701, 1400):
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Deepfake/Deep1_{index}.jpg')

    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    evaluation = predictions[0][0]

    if evaluation >= 0.5:
        count += 1

ratio2 = count / 700

# Final ration of correctly predicted images which values where used in thesis tables
print("Ratio of predicted deepfakes:", ratio1)
print("Ratio of predicted deepfakes:", ratio2)

count = 0
for index in range(1, 701):
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Deepfake/Deep2_{index}.jpg')

    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    evaluation = predictions[0][0]

    if evaluation >= 0.5:
        count += 1

ratio1 = count / 700

count = 0
for index in range(701, 1400):
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Deepfake/Deep2_{index}.jpg')

    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    evaluation = predictions[0][0]

    if evaluation >= 0.5:
        count += 1

ratio2 = count / 700

print("Ratio of predicted deepfakes:", ratio1)
print("Ratio of predicted deepfakes:", ratio2)

count = 0
for index in range(1, 701):
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Deepfake/Deep3_{index}.jpg')

    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    evaluation = predictions[0][0]

    if evaluation >= 0.5:
        count += 1

ratio1 = count / 700
print("Ratio of predicted deepfakes:", ratio1)

count = 0
for index in range(1, 701):
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Deepfake/Deep4_{index}.jpg')

    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    evaluation = predictions[0][0]

    if evaluation >= 0.5:
        count += 1

ratio1 = count / 700

count = 0
for index in range(701, 1400):
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Deepfake/Deep4_{index}.jpg')

    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    evaluation = predictions[0][0]

    if evaluation >= 0.5:
        count += 1

ratio2 = count / 700

print("Ratio of predicted deepfakes:", ratio1)
print("Ratio of predicted deepfakes:", ratio2)

count = 0
for index in range(1, 701):
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Original/Origin_{index}.jpg')

    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    evaluation = predictions[0][0]

    if evaluation >= 0.5:
        count += 1

ratio1 = (700-count) / 700

count = 0
for index in range(701, 1400):
    img1 = Image.open(f'data/Original/Origin_{index}.jpg')
    img2 = Image.open(f'data/Original/Origin_{index}.jpg')

    if img1 is None or img2 is None:
        print(f"Failed to load images for index {index}")
        continue

    img1 = np.array(img1, dtype='float32') / 255.0
    img2 = np.array(img2, dtype='float32') / 255.0
    image_pair = np.stack([img1, img2], axis=0)

    predictions = siamese_model.predict(np.expand_dims(image_pair, axis=0))
    evaluation = predictions[0][0]

    if evaluation >= 0.5:
        count += 1

ratio2 = (700-count) / 700

print("Ratio of predicted originals:", ratio1)
print("Ratio of predicted originals:", ratio2)