from __future__ import absolute_import, division, print_function, unicode_literals
from IPython.display import Image, display
import numpy as np
import os
from os.path import join
from PIL import ImageFile
import pandas as pd
from matplotlib import cm
#import seaborn as sns
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications import ResNet50
import pickle
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

#from keras.applications.Resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from keras.utils import load_img, img_to_array
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import time
import cv2

paths = []
data_type = []
labels = []
train_img_dir_s = "images/garbage"
for dirname, _, filenames in os.walk(train_img_dir_s):
    for filename in filenames:
        if '.jpg' in filename:
            path = dirname + '/' + filename
            paths.append(path)

            if 'garbage' in path:
                labels.append('garbage')
            elif 'noise' in path:
                labels.append('noise')
            else:
                labels.append('N/A')

print(len(paths), len(data_type), len(labels))
data_df = pd.DataFrame({'path': paths,'label': labels})

train_df = data_df[data_df['data_type'] == 'train']
test_df = data_df[data_df['data_type'] == 'test']

# 클래스 불균형 막기 위해 stratify 인자에 레이블에 해당하는 pd.Series 형태로 추가
tr_df, val_df = train_test_split(train_df, stratify=train_df['label'], test_size=0.15, random_state=42)
print('Train:', tr_df.shape, 'Valid:', val_df.shape, 'Test:', test_df.shape)