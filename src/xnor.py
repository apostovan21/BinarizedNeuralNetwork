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
OUTPUT_PATH = 'output/XNOR/'

BATCH_SIZE = 32
EPOCHS = 1
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
    with open(OUTPUT_PATH + 'training_summary/' + test_name + '.txt', 'w') as f:
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
    plt.savefig(OUTPUT_PATH + 'training_plots/' + test_name + '.png')


def get_n_save_test_accuracy(model, X_test, y_test, test_name):
    pred = model.predict(X_test)
    with open(OUTPUT_PATH + 'training_summary/' + test_name + '.txt', 'a') as f:
        with contextlib.redirect_stdout(f):
            print('\nTest Accuracy:')
            print(accuracy_score(y_test, np.argmax(pred, axis=1)))


def post_build_process(model, X_train, X_validation, y_train, y_validation, X_test, y_test, test_name):
    history = compile_n_fit(
        model, X_train, X_validation, y_train, y_validation)

    model.save(OUTPUT_PATH + 'models/' + test_name + '.h5')

    save_summary(model, test_name)
    plot_graphs(history, test_name)

    get_n_save_test_accuracy(model, X_test, y_test, test_name)


def xnor_qconv_mp(X_train, X_validation, y_train, y_validation, X_test, y_test, filter_1, filter_2):
    TEST_NAME = 'XNOR(QConv, MP)/' + str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_' + str(filter_1) + '_3_MP_2_QConv_' + \
        str(filter_2) + '_2_MP_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(filter_1, KERNEL_SIZE_3,
                             filter_2, KERNEL_SIZE_2, NO_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)
    

def xnor_qconv(X_train, X_validation, y_train, y_validation, X_test, y_test, filter_1, filter_2):
    TEST_NAME = 'XNOR(QConv)/' + str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_' + str(filter_1) + '_3_MP_2_QConv_' + \
        str(filter_2) + '_2_MP_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(filter_1, KERNEL_SIZE_3,
                             filter_2, KERNEL_SIZE_2, USE_BN, NO_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def xnor_qconv_mp_enhanced_bn(X_train, X_validation, y_train, y_validation, X_test, y_test, filter_1, filter_2):
    TEST_NAME = 'XNOR(QConv, MP) enhanced (BN)/' + str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_' + str(filter_1) + '_3_MP_2_QConv_' + \
        str(filter_2) + '_2_MP_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(filter_1, KERNEL_SIZE_3,
                             filter_2, KERNEL_SIZE_2, USE_BN, USE_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


def xnor_qconv_modified_bn(X_train, X_validation, y_train, y_validation, X_test, y_test, filter_1, filter_2):
    TEST_NAME = 'XNOR(QConv) modified (BN)/' + str(SIZE) + 'x' + str(SIZE) + '/3_' + str(SIZE) + '_' + str(SIZE) + \
        '_QConv_' + str(filter_1) + '_3_MP_2_QConv_' + \
        str(filter_2) + '_2_MP_2_BN_Dense_43_ep_' + str(EPOCHS)

    model = build_xnor_model(filter_1, KERNEL_SIZE_3,
                             filter_2, KERNEL_SIZE_2, USE_BN, NO_MP)
    post_build_process(model, X_train, X_validation, y_train,
                       y_validation, X_test, y_test, TEST_NAME)


# MAIN CODE

data, labels = get_training_dataset()
X_train, X_validation, y_train, y_validation = split_in_training_n_test(
    data, labels)

X_test, y_test = get_testing_dataset()


xnor_qconv(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_32, FILTER_64)
xnor_qconv(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_64, FILTER_128)
xnor_qconv(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_16, FILTER_32)

xnor_qconv_mp(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_32, FILTER_64)
xnor_qconv_mp(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_64, FILTER_128)
xnor_qconv_mp(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_16, FILTER_32)

xnor_qconv_modified_bn(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_32, FILTER_64)
xnor_qconv_modified_bn(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_64, FILTER_128)
xnor_qconv_modified_bn(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_16, FILTER_32)

xnor_qconv_mp_enhanced_bn(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_32, FILTER_64)
xnor_qconv_mp_enhanced_bn(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_64, FILTER_128)
xnor_qconv_mp_enhanced_bn(X_train, X_validation,
                          y_train, y_validation, X_test, y_test, FILTER_16, FILTER_32)
