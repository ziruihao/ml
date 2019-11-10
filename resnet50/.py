import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50

from IPython.display import Image, display

from os.path import join
import json

y_image_dir = './data/test/hot_dog'

y_img_paths = [join(y_image_dir, filename) for filename in ['1000288.jpg', '127117.jpg']]

n_image_dir = './data/test/not_hot_dog'

n_img_paths = [join(n_image_dir, filename) for filename in ['823536.jpg', '99890.jpg']]

img_paths = y_img_paths + n_img_paths

image_size = 224

for img in img_paths:
    display(Image(img_paths))

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return output

def decode_predictions(preds, top=5, class_list_path='./model/imagenet_class_index.json'):
  """Decodes the prediction of an ImageNet model.
  Arguments:
      preds: Numpy tensor encoding a batch of predictions.
      top: integer, how many top-guesses to return.
      class_list_path: Path to the canonical imagenet_class_index.json file
  Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.
  Raises:
      ValueError: in case of invalid shape of the `pred` array
          (must be 2D).
  """
  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  CLASS_INDEX = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results

my_model = ResNet50(weights='./model/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)

# most_likely_labels = decode_predictions(preds, top=3)

# for i, img_path in enumerate(img_paths):
#     display(Image(img_path))
#     print(most_likely_labels[i])