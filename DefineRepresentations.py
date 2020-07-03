import tensorflow as tf
import numpy as np

"""defining content and style representations"""

# loading vgg without classification head and listing layer names
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

""" code to print layer names to choose from"""
'''
print()
for layer in vgg.layers:
    print(layer.name)
'''

# choosing intermediate layers from the network to represent the style and content of the image

content_layers = ['block5_conv1']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

