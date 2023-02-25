from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
import contextlib
import larq as lq

# CONSTANTS
CLASSES = 43
SIZE = 30
IMG_RESIZE = (SIZE, SIZE)
CURRENT_PATH = os.getcwd()   

BATCH_SIZE = 32
EPOCHS = 500
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

FILTER_128 = 128
FILTER_64 = 64
FILTER_32 = 32
FILTER_16 = 16
KERNEL_SIZE_5 = (5, 5)
KERNEL_SIZE_3 = (3, 3)
KERNEL_SIZE_2 = (2, 2)

USE_BN = True
USE_MP = True
NO_BN = False
NO_MP = False
####################


def get_training_dataset():
    data = []
    labels = []
    # Retrieving the images and their labels
    for i in range(CLASSES):
        path = os.path.join(
            CURRENT_PATH, 'datasets/GTSRB_dataset/Train', str(i))
        # print(path)
        images = os.listdir(path)

        for a in images:
            #print(path + '\\' + a)

            try:
                image = Image.open(path + '/' + a)
                image = image.resize(IMG_RESIZE)
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    # Converting lists into numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    print(data.shape, labels.shape)
    return data, labels


def get_testing_dataset():
    ds = pd.read_csv('datasets/GTSRB_dataset/Test.csv')
    y_test = ds["ClassId"].values
    imgs = ds["Path"].values

    data = []
    for img in imgs:
        image = Image.open('datasets/GTSRB_dataset/' + img)
        image = image.resize(IMG_RESIZE)
        data.append(np.array(image))
    X_test = np.array(data)
    
    return X_test, y_test

def split_in_training_n_test(data, labels):
    X_train, X_validation, y_train, y_validation = train_test_split(
        data, labels, test_size=0.2, random_state=42)
    print(X_train.shape, X_validation.shape, y_train.shape, y_validation.shape)
    y_train = to_categorical(y_train, CLASSES)
    y_validation = to_categorical(y_validation, CLASSES)
    return X_train, X_validation, y_train, y_validation

def build_xnor_model(filter1, kernel_size1, filter2, kernel_size2, use_batchnormalization, use_maxpooling):
    # Building the model
    model = tf.keras.models.Sequential()

    # In the first layer we only quantize the weights and not the input
    model.add(lq.layers.QuantConv2D(filter1, kernel_size1,
                                    kernel_quantizer="ste_sign",
                                    kernel_constraint="weight_clip",
                                    use_bias=False,
                                    input_shape=(SIZE, SIZE, 3)))
    if use_maxpooling:
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # Block 2
    model.add(lq.layers.QuantConv2D(
        filter2, kernel_size2, use_bias=False, **kwargs))
    if use_maxpooling:
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    if use_batchnormalization:
        model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(tf.keras.layers.Flatten())
    # Output layer
    model.add(lq.layers.QuantDense(CLASSES, use_bias=False, **kwargs))
    model.add(tf.keras.layers.Activation("softmax"))

    return model

def compile_n_fit(model, X_train, X_validation, y_train, y_validation):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                        epochs=EPOCHS, validation_data=(X_validation, y_validation))
    return history

def save_summary(model, test_name):
    with open('xnor/training_summary/' + test_name + '.txt', 'w') as f:
        with contextlib.redirect_stdout(f):
            lq.models.summary(model)
        
def plot_graphs(history, test_name):
    df = pd.DataFrame(history.history).rename_axis(
        'epoch').reset_index().melt(id_vars=['epoch'])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, mtr in zip(axes.flat, ['loss', 'accuracy']):
        ax.set_title(f'{mtr.title()} Plot')
        dfTmp = df[df['variable'].str.contains(mtr)]
        sns.lineplot(data=dfTmp, x='epoch', y='value', hue='variable', ax=ax)
    fig.tight_layout()
    # plt.show()
    plt.savefig('xnor/training_plots/'+ test_name + '.png')

def get_n_save_test_accuracy(model, X_test, y_test, test_name):
    pred = model.predict(X_test)
    with open('xnor/training_summary/' + test_name + '.txt', 'a') as f:
        with contextlib.redirect_stdout(f):
            print('\nTest Accuracy:')
            print(accuracy_score(y_test, np.argmax(pred, axis=1)))
 
def post_build_process(model, X_train, X_validation, y_train, y_validation, X_test, y_test, test_name):
    history = compile_n_fit(
        model, X_train, X_validation, y_train, y_validation)

    model.save('xnor/models/' + test_name + '.h5')

    save_summary(model, test_name)
    plot_graphs(history, test_name)

    get_n_save_test_accuracy(model, X_test, y_test, test_name)
    
## Case 1
def case1_xnor(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case1/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_3_MP_2_QConv_64_2_MP_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_32, KERNEL_SIZE_3,
                            FILTER_64, KERNEL_SIZE_2, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def case1_xnor_no_mp(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case1/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_3_QConv_64_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_32, KERNEL_SIZE_3,
                             FILTER_64, KERNEL_SIZE_2, USE_BN, NO_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    

def case1_xnor_no_mp_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case1/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_3_QConv_64_2_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_32, KERNEL_SIZE_3,
                             FILTER_64, KERNEL_SIZE_2, NO_BN, NO_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    

def case1_xnor_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case1/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_32_3_MP_2_QConv_64_2_MP_2_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_32, KERNEL_SIZE_3,
                             FILTER_64, KERNEL_SIZE_2, NO_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    
## Case 2 ##


def case2_xnor(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case2/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_64_3_MP_2_QConv_128_2_MP_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_64, KERNEL_SIZE_3,
                             FILTER_128, KERNEL_SIZE_2, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def case2_xnor_no_mp(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case2/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_64_3_QConv_128_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_64, KERNEL_SIZE_3,
                             FILTER_128, KERNEL_SIZE_2, USE_BN, NO_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def case2_xnor_no_mp_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case2/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_64_3_QConv_128_2_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_64, KERNEL_SIZE_3,
                             FILTER_128, KERNEL_SIZE_2, NO_BN, NO_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def case2_xnor_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case2/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_64_3_MP_2_QConv_128_2_MP_2_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_64, KERNEL_SIZE_3,
                             FILTER_128, KERNEL_SIZE_2, NO_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    
## Case 3


def case3_xnor(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case3/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_16_3_MP_2_QConv_32_2_MP_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_16, KERNEL_SIZE_3,
                             FILTER_32, KERNEL_SIZE_2, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def case3_xnor_no_mp(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case3/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_16_3_QConv_32_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_16, KERNEL_SIZE_3,
                             FILTER_32, KERNEL_SIZE_2, USE_BN, NO_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def case3_xnor_no_mp_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case3/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_16_3_QConv_32_2_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_16, KERNEL_SIZE_3,
                             FILTER_32, KERNEL_SIZE_2, NO_BN, NO_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def case3_xnor_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test):
    TEST_NAME = 'case3/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_16_3_MP_2_QConv_32_2_MP_2_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(FILTER_16, KERNEL_SIZE_3,
                             FILTER_32, KERNEL_SIZE_2, NO_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)

# MAIN CODE


data, labels = get_training_dataset()
X_train, X_validation, y_train, y_validation = split_in_training_n_test(
    data, labels)

X_test, y_test = get_testing_dataset()

print("\n!!!!! CASE 1 !!!!!\n")
case1_xnor(X_train, X_validation, y_train, y_validation, X_test, y_test)
case1_xnor_no_mp(X_train, X_validation, y_train, y_validation, X_test, y_test)
case1_xnor_no_mp_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test)
case1_xnor_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test)
# print("\n!!!!! CASE 2 !!!!!\n")
# case2_xnor(X_train, X_validation, y_train, y_validation, X_test, y_test)
# case2_xnor_no_mp(X_train, X_validation, y_train, y_validation, X_test, y_test)
# case2_xnor_no_mp_no_bn(X_train, X_validation, y_train,
#                        y_validation, X_test, y_test)
# case2_xnor_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test)
# print("\n!!!!! CASE 3 !!!!!\n")
# case3_xnor(X_train, X_validation, y_train, y_validation, X_test, y_test)
# case3_xnor_no_mp(X_train, X_validation, y_train, y_validation, X_test, y_test)
# case3_xnor_no_mp_no_bn(X_train, X_validation, y_train,
#                        y_validation, X_test, y_test)
# case3_xnor_no_bn(X_train, X_validation, y_train, y_validation, X_test, y_test)
