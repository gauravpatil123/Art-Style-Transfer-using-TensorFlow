import PIL.Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['axes.grid'] = False

# image directory paths
content_image_path = "Images/source/content.jpg"
style_image_path = "Images/source/style.jpg"

""" helper functions """

def tensorToImage(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# function to load image with set max dimention
def loadImage(path):
    maxdim = 512
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    longdim = max(shape)
    scale = maxdim / longdim

    newShape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, newShape)
    image = image[tf.newaxis, :]
    return image

# function to display images
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

""" displaying images"""
content_img = loadImage(content_image_path)
style_img = loadImage(style_image_path)

plt.subplot(1, 2, 1)
imshow(content_img, "Content Image")

plt.subplot(1, 2, 2)
imshow(style_img, "Style Image")

plt.savefig("Images/outputs/resaled_content_and_style_images.png")
#plt.show()
