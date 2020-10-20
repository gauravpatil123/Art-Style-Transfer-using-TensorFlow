"""
DefineRepresentations:
    Initializes Class RepresentationLayers
"""

import tensorflow as tf
import numpy as np

class RepresentationLayers:

    """
    Class to initialize content and style layers 
    """

    def __init__(self):
        """
        Initializes the content and style layers from the VGG model
        Initializes the length of chosen content and style layers list
        """
        self.content_layers = ['block5_conv1']
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)

    def __call__(self):
        """
        Returns content and style layers
        """
        return self.content_layers, self.style_layers

    def get_num_style_layers(self):
        """
        Returns number of style layers
        """
        return self.num_style_layers

    def get_num_content_layers(self):
        """
        Returns number of content layers
        """
        return self.num_content_layers
