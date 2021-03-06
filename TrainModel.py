"""
TrainModel:
    Trains the model and saves the converted image
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['axes.grid'] = False
import PIL.Image
import time
import functools
import IPython.display as display
import ImagePreprocessing as IP
import DefineRepresentations as DR
import BuildModel as BM
import logging

# Initializing objects
CONTENT_IMAGE = IP.content_img
STYLE_IMAGE = IP.style_img
LAYER_REP = DR.RepresentationLayers()
CONTENT_LAYERS, STYLE_LAYERS = LAYER_REP()
NUM_CONTENT_LAYERS = LAYER_REP.get_num_content_layers()
NUM_STYLE_LAYERS = LAYER_REP.get_num_style_layers()
EXTRACTOR = BM.StyleContentExtraction(STYLE_LAYERS, CONTENT_LAYERS, STYLE_IMAGE, CONTENT_IMAGE, True, True)

# logging configuration
logging.basicConfig(format="%(message)s", level=logging.INFO)

"""Running Gradient Descent"""
# setting style and content parameter targets
style_targets = EXTRACTOR(STYLE_IMAGE)['style']
content_targets = EXTRACTOR(CONTENT_IMAGE)['content']

# defining tf.Variable to contain and optimize the image
image = tf.Variable(CONTENT_IMAGE)

# function to keep pixel values between 0 and 1
def regularize(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# choosing optimizer for SGD and setting hyperparameters
optimizer = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.99, epsilon=1e-1)

# configuring weights for content and style
STYLE_WEIGHT = 5e-2
CONTENT_WEIGHT = 1e4

# loss function for style and content features
def losses(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= STYLE_WEIGHT / NUM_STYLE_LAYERS
    
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                              for name in content_outputs.keys()])

    content_loss *= CONTENT_WEIGHT / NUM_CONTENT_LAYERS
    loss = style_loss + content_loss
    return loss

# setting weight total variation loss optimization
TOTAL_VARIATION_WEIGHT = 30

# using Gradient tape to update image
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = EXTRACTOR(image)
        loss = losses(outputs)
        loss += TOTAL_VARIATION_WEIGHT*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(regularize(image))

# re-intializing the optimization variable
image = tf.Variable(CONTENT_IMAGE)

# optimization
start = time.time()

# training parameters
epochs = 25
steps_per_epoch = 100

step = 0
start_log = "Starting optimization loop:"
logging.info(start_log)
for n in range(epochs):
    epoch_log = "epoch = " + str(n + 1)
    logging.info(epoch_log)
    for m in range(steps_per_epoch):
        step += 1
        if step % 10 == 0:
            step_log = "Step = " + str(step)
            logging.info(step_log)
        train_step(image)
    file_name = "Images/outputs/epoch_" + str(n + 1) + ".png"
    IP.tensorToImage(image).save(file_name)
    display.clear_output(wait=True)
    display.display(IP.tensorToImage(image).show())
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))

file_name = 'Images/outputs/stylized-image.png'
IP.tensorToImage(image).save(file_name)
