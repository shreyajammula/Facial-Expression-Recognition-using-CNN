# Facial Expression Recognition using CNN

## Overview
This project involves training a Convolutional Neural Network (CNN) for facial expression recognition using the FER2013 dataset. The trained model can predict emotions (Angry, Fear, Happy, Sad, Surprise) from facial images.

## Project Structure
- `Model_train.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, and saving the model.
- `Model_test.ipynb`: Jupyter notebook with code to test the trained model using new data.
- `face_cnn_model_architecture.json`: JSON file containing the architecture of the trained model.
- `face_cnn_model_weights.h5`: HDF5 file containing the weights of the trained model.

## Dataset:
  Download Your Dataset from Here
- [Kaggle FER2018](https://www.kaggle.com/datasets/ashishpatel26/fer2018)

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/AjayK47/Face-Emotion-Detection_CNN.git
   ``` 
2.  Run Model_train.ipynb to train the CNN model and save the model architecture and weights

3.  If you just want to test out model just Run Model_test file 

## Dependencies

1. Numpy
2. Pandas
3. tensorflow
4. scikit-learn
5. Pillow

## Acknowledgements
1. -[Fer2018 Dataset is provided from Kaggle](https://www.kaggle.com/datasets/ashishpatel26/fer2018)
   

