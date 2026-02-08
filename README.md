# 🚦 Traffic Sign Classification (CNN – TensorFlow/Keras)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?logo=keras&logoColor=white)](https://keras.io)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-green?logo=opencv&logoColor=white)](https://opencv.org)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-lightblue?logo=numpy&logoColor=white)](https://numpy.org)


## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) for traffic sign image classification using TensorFlow/Keras.
The model is trained on a dataset of traffic sign images and predicts one of 43 traffic sign classes.


## 🚀 Features

| 🧠 **Deep Learning Model** | 🖼 **Image Processing** | 📊 **Training Optimization** |
|---|---|---|
| Custom CNN Architecture | Image Normalization | Early Stopping |
| 43-Class Classification | Data Augmentation | Validation Monitoring |
| Softmax Multi-Class Output | 32×32 RGB Images | Adam Optimizer |



## 📊 Dataset Overview

| Property | Value |
|---|---|
| Classes | 43 |
| Training Images | 58,510 |
| Validation Images | 14,629 |
| Image Size | 32 × 32 |
| Channels | RGB (3) |




## 🛠 Tech Stack

```bash
Deep Learning: TensorFlow + Keras
Data Processing: NumPy + Pandas
Visualization: Matplotlib
Image Processing: OpenCV (Optional)
Training: ImageDataGenerator (Augmentation)
```




## 📁 Project Structure

```bash
traffic-sign-classifier/
├── train/ # Training images (43 class folders)
├── val/ # Validation images (43 class folders)
├── labels.csv # Class label mapping
├── model.py # CNN model architecture
├── train.py # Training script
├── README.md
└── requirements.txt
```



## 🧠 Model Architecture

```bash
Conv2D(16) → MaxPool
Conv2D(32) → MaxPool
Conv2D(64)
Flatten
Dense(512)
Dense(256, tanh)
Dense(128)
Dense(43, softmax)
```

## 🎯 Model Output
```bash
Input → Traffic Sign Image (32x32 RGB)
Output → Class Probability Distribution (43 Classes)
Prediction → Highest Probability Class
```

## 🤝 **Contributing**

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open Pull Request


## 🙏 **Acknowledgments**
- [TensorFlow](https://tensorflow.org) - Deep learning framework for model training and deployment  
- [Keras](https://keras.io) - High-level neural network API for building CNN architecture  
- [NumPy](https://numpy.org) - Efficient numerical computation and array processing  
- [OpenCV](https://opencv.org) - Image processing utilities for computer vision tasks  
- [Matplotlib](https://matplotlib.org) - Training visualization and performance plots  
- Traffic Sign Dataset Providers - High-quality labeled traffic sign image dataset for training and validation  



<div align="center">

**⭐ Star this repo if it helped you!**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Made with TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Made with Keras](https://img.shields.io/badge/Made%20with-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)

</div>
