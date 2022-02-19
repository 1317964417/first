import pickle
import numpy as np
import cv2
import os

# cifar10下载地址：http://www.cs.toronto.edu/~kriz/cifar.html

PATH_DATA = './cifar-10-python/cifar-10-batches-py'
PATH_TRAIN = PATH_DATA + '/train'
PATH_TEST = PATH_DATA + '/test'
PATH_CLASS = PATH_DATA + '/batches.meta'


def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_
    # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])


def unpack_classname(datapath):
    raw_data = unpickle(datapath)
    raw_data = raw_data[b'label_names']
    length = np.linspace(0, len(raw_data), len(raw_data), endpoint=False)
    return {int(i): str(j, encoding='utf-8') for i, j in zip(length, raw_data)}


def process_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unpack_image(label_dict, data_paths, save_path):
    for path in data_paths:
        raw_data = unpickle(path)
        for img, img_name, label in zip(raw_data[b'data'], raw_data[b'filenames'], raw_data[b'labels']):
            img = img.reshape(3, 32, 32).transpose(1, 2, 0)
            label_path = save_path + '/' + label_dict[label]
            process_path(label_path)
            img_path = os.path.join(label_path, str(img_name, encoding='utf-8'))
            cv2.imwrite(img_path, img)


def process_cifar_data():
    files = os.listdir(PATH_DATA)
    process_path(PATH_TRAIN)
    process_path(PATH_TEST)
    train_paths = []
    test_paths = []
    for file in files:
        if file.endswith(('0', '1', '2', '3', '4', '5')):
            train_paths.append(file)
        if file.endswith('batch'):
            test_paths.append(file)
    path_train = [os.path.join(PATH_DATA, x) for x in train_paths]
    path_test = [os.path.join(PATH_DATA, x) for x in test_paths]
    label_dict = unpack_classname(PATH_CLASS)
    unpack_image(label_dict, path_train, PATH_TRAIN)
    print("Finished to unpikle the train data!")
    unpack_image(label_dict, path_test, PATH_TEST)
    print("Finished to unpikle the test data!")


if __name__ == '__main__':
    process_cifar_data()
