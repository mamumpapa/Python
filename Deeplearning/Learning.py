from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
from os.path import join
import pandas as pd
import pickle
from keras.applications import DenseNet201
from keras.applications import ResNet50

from keras.applications.resnet import preprocess_input
from keras.utils import load_img, img_to_array
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import time


start=time.time()
train_img_dir_s = "images/garbage"
all_train_img_paths_s = [join(train_img_dir_s,filename) for filename in os.listdir(train_img_dir_s)]

train_img_paths, test_img_paths_trashcan = train_test_split(all_train_img_paths_s, test_size=0.00001, random_state=42)

natural_images_path="images/"
test_img_paths_no_trashcan = []

for d in [d for d in os.listdir("images/") if d!="garbage" and d!="test" and d!="test2" and d!="test3" and d!="predict_image" and d!="test4"]:
    test_img_dir_na = natural_images_path + d
    test_img_paths_no_trashcan.append([join(test_img_dir_na,filename) for filename in os.listdir(test_img_dir_na)])

test_img_paths_no_trashcan_flat = [item for sublist in test_img_paths_no_trashcan for item in sublist]
test_img_paths_no_trashcan, _ = train_test_split(test_img_paths_no_trashcan_flat, test_size = 0.00001, random_state = 42)

def natural_img_dir(image_path):
    path_regex = r"images\/(\w*)"
    if 'images' in image_path:
        return re.findall(path_regex,image_path,re.MULTILINE)[0].strip()
    else:
        return 'trashcan'

all_test_paths = test_img_paths_trashcan + test_img_paths_no_trashcan
test_path_df = pd.DataFrame({
    'path': all_test_paths,
    'is_trashcan': ['1' if path in test_img_paths_trashcan else '0' for path in all_test_paths]
})

test_path_df = shuffle(test_path_df, random_state=0).reset_index(drop=True)

test_path_df['image_type'] = test_path_df['path'].apply(lambda x: natural_img_dir(x))
all_test_paths = test_path_df['path'].tolist()


test_num=test_path_df[['is_trashcan']].to_numpy()



image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(img_array)

X_train = read_and_prep_images(all_train_img_paths_s)
X_test = read_and_prep_images(all_test_paths)


new_val_path2="images/test3"
new_val_path = [join(new_val_path2,filename) for filename in os.listdir(new_val_path2)]
new_val=read_and_prep_images(new_val_path)

def extract(X):
    Densenet_model = DenseNet201(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False)
    features_array = Densenet_model.predict(X)
    return features_array

X_train=extract(X_train)
X_test=extract(X_test)
new_val=extract(new_val)

X_train_re = np.reshape(X_train,(285,-1))

X_test_re = np.reshape(X_test,(240,-1))
new_val_re=np.reshape(new_val,(3,-1))

ss = StandardScaler()
ss.fit(X_train_re)
X_train_re = ss.transform(X_train_re)
X_test_re = ss.transform(X_test_re)
new_val_re=ss.transform(new_val_re)

pca = PCA(n_components=159, whiten=True)
pca = pca.fit(X_train_re)
X_train_re = pca.transform(X_train_re)
X_test_re = pca.transform(X_test_re)
new_val_re=pca.transform(new_val_re)

# Train classifier and obtain predictions for OC-SVM
if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)  # Obtained using grid search

if_clf.fit(X_train_re)

with open('pca.h5','wb') as f:
    pickle.dump(pca,f)
with open('ss.h5','wb') as f:
    pickle.dump(ss,f)
with open('if_clf.h5','wb') as f:
    pickle.dump(if_clf,f)

if_preds = if_clf.predict(X_test_re)

tmp=0

test_num=test_path_df[['is_trashcan']].to_numpy()
for i, j in zip(test_num.astype(int), if_preds):
    if(i==j+1):
        tmp+=1

print("테스트 데이터셋에서 정확도:", tmp/if_preds.shape[0])
