# Binarized Neural Network

## Summary
This work presents a bottom-up approach for architecturing BNNs by studying characteristics of the constituent layers. These constituent layers (binarized convolutional layers, max pooling, batch normalization, fully connected layers) are studied in various combinations and with different values of kernel size, number of filters and of neurons by using the German Traffic Sign Recognition Benchmark (GTSRB) for training. As a result, we propose BNNs architectures which achieve more than 90% for GTSRB (the maximum is 96.45%) and an average greater than 80% (the maximum is 88.99%) considering also the Belgian and Chinese datasets for testing. The number of parameters of these architectures varies from 100k to less than 2M.

## Requirements
You have to install:
- Tensorflow
```
pip install tensorflow==2.10.0
```
- Keras
```
pip install keras==2.10.0
```
- Larq
```
pip install larq==0.12.2
```


## Datasets
- [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?datasetId=82373&language=Python)
- [Belgium](https://www.kaggle.com/datasets/shazaelmorsh/trafficsigns)
- [Chinese](https://www.kaggle.com/datasets/dmitryyemelyanov/chinese-traffic-signs)

However, for Belgium and Chinese datasets we have to modify the order of the classes in order to correspond to the **GTSRB** dataset. You can download the datasets from the [drive](https://drive.google.com/drive/folders/1OMHjWpLJ9DnLBfSyTKLJ-fFLOzsNI9Qh).

In the workspace you should have following folders:
- datasets
  - Belgian_dataset
  - Chinese_dataset
  - GTSRB_dataset


## How to run the training scripts
Depending on which architecture you want to train, you can choose to run a certain script:
  - [xnor.py](https://github.com/apostovan21/BinarizedNeuralNetwork/blob/master/src/xnor.py)
  - [3QConv.py](https://github.com/apostovan21/BinarizedNeuralNetwork/blob/master/src/3QConv.py)
  - extra architectures
    - [3QConv_v2.py](https://github.com/apostovan21/BinarizedNeuralNetwork/blob/master/src/3QConv_v2.py)
    - [3QConv_v3.py](https://github.com/apostovan21/BinarizedNeuralNetwork/blob/master/src/3QConv_v3.py)
    - [3QConv_v4.py](https://github.com/apostovan21/BinarizedNeuralNetwork/blob/master/src/3QConv_v4.py)

Before runing the scripts, make sure you have following folders created in your workspace:
- datasets
  - Belgian_dataset
  - Chinese_dataset
  - GTSRB_dataset
- output
  - 3QConv
    - confusion_matrix
      - 30x30
      - 48x48
      - 64x64
    - models
      - 30x30
      - 48x48
      - 64x64
    - training_plots
      - 30x30
      - 48x48
      - 64x64
    - training_summary
      - 30x30
      - 48x48
      - 64x64
  - extra_architectures
    - 3QConv_v2
      - *Same as for* **3QConv**.
    - 3QConv_v3
      - *Same as for* **3QConv**.
    - 3QConv_v4
      - *Same as for* **3QConv**.
  - XNOR
    - *Same as for* **3QConv**.

Each script has some constants defined which you can change according to the training you want to make.
- `SIZE` by default if `30`. You can change to either `48` or `64`.
- `EPOCHS` by default are `30`. You can change it as you want.

At the end of each script there is the `main` code which calls the functions for training. If you don't want to train all variations of architectures, you can comment a certain training.

## How to run the predition scripts
First make sure you have run the training for desire architecture and you have the model and training summary saved.
Next, edit the constants from [predition.py](https://github.com/apostovan21/BinarizedNeuralNetwork/blob/master/src/prediction.py):

- `SIZE`
- `OUTPUT_PATH`
- `test_name`

Run the script. It will update the training summary and will create confusion matrixes.

## Output Results
If you want to check our out results, you can check them [here](https://drive.google.com/drive/folders/1y2n7V7nr0tBQQSS8DhL1shnjgYUIrZ9Z).

## License

TODO: Add
