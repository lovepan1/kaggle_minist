import pandas as pd
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

def extract_images_and_labels(dataset, validation = False):
    images = dataset[:, 1:].reshape(-1, 28, 28, 1)
    labels_dense = dataset[:, 0]
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * 10
    labels_one_hot = np.zeros((num_labels, 10))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    if validation:
        num_images = images.shape[0]
        divider = num_images - 200
        return images[:divider], labels_one_hot[:divider], images[divider+1:], labels_one_hot[divider+1:]
    else:
        return images, labels_one_hot

def extract_images(dataset):
    return dataset.reshape(-1, 28*28)

train_data_file = 'train.csv'
test_data_file = 'test.csv'

train_data = pd.read_csv(train_data_file).as_matrix().astype(np.uint8)
test_data = pd.read_csv(test_data_file).as_matrix().astype(np.uint8)

train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
test_images = extract_images(test_data)

#创建DataSet对象
train = DataSet(train_images,
                train_labels,
                dtype = np.float32,
                reshape = True)

validation = DataSet(val_images,
                val_labels,
                dtype = np.float32,
                reshape = True)

test = test_images
