from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import os
import collections
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import contextlib

import larq as lq

OUTPUT_PATH = 'output/3QConv/'
SIZE = 30
IMG_RESIZE = (SIZE, SIZE)

classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

classesAbv = {0: 'Sp L (20km/h)',
              1: 'Sp L (30km/h)',
              2: 'Sp L (50km/h)',
              3: 'Sp L (60km/h)',
              4: 'Sp L (70km/h)',
              5: 'Sp L (80km/h)',
              6: 'End of Sp L (80km/h)',
              7: 'Sp L (100km/h)',
              8: 'Sp L (120km/h)',
              9: 'No passing',
              10: 'NoPass > 3.5 tons',
              11: 'Right-of-way ...',
              12: 'Priority road',
              13: 'Yield',
              14: 'Stop',
              15: 'No vehicles',
              16: 'Veh > 3.5t prohibited',
              17: 'No entry',
              18: 'General caution',
              19: 'Dangerous curve L',
              20: 'Dangerous curve R',
              21: 'Double curve',
              22: 'Bumpy road',
              23: 'Slippery road',
              24: 'Road narrows R',
              25: 'Road work',
              26: 'Traffic signals',
              27: 'Pedestrians',
              28: 'Children crossing',
              29: 'Bicycles crossing',
              30: 'Beware of ice/snow',
              31: 'Wild an. crossing',
              32: 'End Sp + pass limits',
              33: 'Turn right ahead',
              34: 'Turn left ahead',
              35: 'Ahead only',
              36: 'Go straight or R',
              37: 'Go straight or L',
              38: 'Keep right',
              39: 'Keep left',
              40: 'Roundabout M',
              41: 'End of no passing',
              42: 'End NoPass V>3.5t'}


def get_dataset(path_to_ds, class_id_col_name, path_to_img_col_name, path_to_img):
    ds = pd.read_csv(path_to_ds)

    y_test = ds[class_id_col_name].values
    imgs = ds[path_to_img_col_name].values

    data = []

    for img in imgs:
        image = Image.open(path_to_img + img)
        image = image.resize(IMG_RESIZE)
        data.append(np.array(image))

    X_test = np.array(data)
    return X_test, y_test


def get_n_save_accuracy(X_test, y_test, test_name, dataset_name):
    model = tf.keras.models.load_model(
        OUTPUT_PATH + 'models/' + test_name + '.h5')
    pred = model.predict(X_test)
    y_pred = np.argmax(pred, axis=1)
    acc = accuracy_score(y_test, np.argmax(pred, axis=1))

    with open(OUTPUT_PATH + 'training_summary/' + test_name + '.txt', 'a') as f:
        with contextlib.redirect_stdout(f):
            print('\n' + dataset_name + ' dataset accuracy: ' + str(acc))
    return y_pred


def get_n_save_confusion_matrix(X_test, y_test, y_pred, test_name, dataset_name, should_delete_rows=False):
    cm = confusion_matrix(y_test, y_pred, normalize='true')  # normalize='true'

    disp_labels = classes.values()
    if(should_delete_rows):
        newClasses = {}
        for label in y_test:
            newClasses[label] = classes[label+1]
        rows = cm.any(axis=1)
        toDelete = []
        for i in range(0, len(rows)):
            if(rows[i] == False):
                toDelete.append(i)
        cm = np.delete(cm, toDelete, 0)
        cm = np.delete(cm, toDelete, 1)
        sortedClasses = collections.OrderedDict(sorted(newClasses.items()))
        disp_labels = sortedClasses.values()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=disp_labels)
    fig, ax = plt.subplots(figsize=(30, 30))
    if dataset_name == 'German':
        disp.plot(cmap=plt.cm.Blues,
                  ax=ax, values_format='.1g')
    elif dataset_name == 'Chinese':
        disp.plot(cmap=plt.cm.Greens,
                  ax=ax, values_format='.1g')
    elif dataset_name == 'Belgian':
        disp.plot(cmap=plt.cm.RdPu,
                  ax=ax, values_format='.1g')
    plt.xticks(rotation=90)
    plt.title(dataset_name + '_' + test_name)
    plt.savefig(OUTPUT_PATH + 'confusion_matrix/' +
                test_name + '_' + dataset_name + '.png')


def readDsBelgianByClass(belgianClass):
    name = '00000'
    if belgianClass > 9:
        name = '000' + str(belgianClass)
    else:
        name = '0000' + str(belgianClass)
    y_test = pd.read_csv('datasets/Belgian_dataset/BelgiumTSC_Training/Training/' +
                         name + '/newGT-' + name + '.csv')
    return y_test, name


def getLabelsAndImagesBelgian():
    belgianClasses = [31, 17, 61, 19, 21, 28, 25, 22, 13,
                      3, 4, 5, 6, 0, 2, 16, 10, 11, 7, 8,
                      9, 34, 36, 37]
    labels = []
    data = []
    for bClass in belgianClasses:
        y_test, name = readDsBelgianByClass(bClass)
        newClass = y_test["ClassId"].values[0]
        imgs = y_test["Filename"].values
        for img in imgs:
            image = Image.open(
                'datasets/Belgian_dataset/BelgiumTSC_Training/Training/' + name + '/' + img)
            image = image.resize(IMG_RESIZE)
            data.append(np.array(image))
            labels.append(newClass)
            X_test = np.array(data)
    return X_test, labels


def predict_all(german_x_test, german_y_test, chinese_x_test, chinese_y_test, belgium_x_test, belgium_y_test, test_name):
    # German
    german_y_pred = get_n_save_accuracy(
        german_x_test, german_y_test, test_name, 'German')
    get_n_save_confusion_matrix(
        german_x_test, german_y_test, german_y_pred, test_name, 'German')
    # Chinese
    chinese_y_pred = get_n_save_accuracy(
        chinese_x_test, chinese_y_test, test_name, 'Chinese')
    get_n_save_confusion_matrix(
        chinese_x_test, chinese_y_test, chinese_y_pred, test_name, 'Chinese', True)
    # Belgium
    belgium_y_pred = get_n_save_accuracy(
        belgium_x_test, belgium_y_test, test_name, 'Belgian')
    get_n_save_confusion_matrix(
        belgium_x_test, belgium_y_test, belgium_y_pred, test_name, 'Belgian', True)


#########################
# MAIN CODE

german_x_test, german_y_test = get_dataset(
    'datasets/GTSRB_dataset/Test.csv', 'ClassId', 'Path', 'datasets/GTSRB_dataset/')
chinese_x_test, chinese_y_test = get_dataset(
    'datasets/Chinese_dataset/annotations_v5.csv', 'category', 'file_name', 'datasets/Chinese_dataset/images/')
belgium_x_test, belgium_y_test = getLabelsAndImagesBelgian()

test_name = '3_' + str(SIZE) + '_' + str(SIZE) + \
    '_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_43_ep_30'
predict_all(german_x_test, german_y_test, chinese_x_test,
            chinese_y_test, belgium_x_test, belgium_y_test, test_name)
