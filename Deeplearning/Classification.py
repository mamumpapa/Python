from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
from os.path import join

from keras.applications.resnet import preprocess_input
from keras.utils import load_img, img_to_array
import pickle
from keras.applications import DenseNet201


def read_and_prep_images(img_paths, img_height=224, img_width=224):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(img_array)

def extract(X):
    Densenet_model = DenseNet201(input_shape=(224,224 , 3), weights='imagenet', include_top=False)
    features_array = Densenet_model.predict(X)
    return features_array

def model(path):
    new_val_path = [path]
    print(new_val_path)
    new_val = read_and_prep_images(new_val_path)

    new_val = extract(new_val)

    new_val_re = np.reshape(new_val, (1, -1))

    with open("ss.h5", "rb") as fr:
        ss = pickle.load(fr)
    new_val_re = ss.transform(new_val_re)

    with open("pca.h5", "rb") as fr:
        pca = pickle.load(fr)
    new_val_re = pca.transform(new_val_re)

    with open("if_clf.h5", "rb") as fr:
        if_clf = pickle.load(fr)
    if_preds2 = if_clf.predict(new_val_re)
    print(if_preds2[0])
    return if_preds2[0]

path="img/img.jpg"
model(path)