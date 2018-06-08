import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import numpy as np

def get_image(image_path, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def transform(image):
  return np.array(image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.
  #return images

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)