# Twitter-Sentiment-Analysis-using-CNN-LSTM-and-Sentiment140-dataset

## Overview

This project implements a comprehensive sentiment analysis pipeline for Twitter data using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The objective is to classify tweets as **positive** or **negative** using both classic machine learning and deep learning (CNN-LSTM) models.  

## Workflow

1. **Data Loading**
   - Download Sentiment140 and load into a pandas DataFrame

2. **Preprocessing**
   - Drop unused columns
   - Demojize tweets
   - Expand contractions and slangs
   - Clean text (lowercase, remove punctuation, etc.)
   - Tokenize, remove stopwords, lemmatize
   - Spelling correction
   - Hashtag normalization

3. **Vectorization & Embedding**
   - TF-IDF for classic ML
   - Tokenizer + padded sequences for LSTM/CNN-LSTM

4. **Model Training**
   - Train classic ML models for baseline
   - Build and train a CNN-LSTM model with Keras

5. **Evaluation**
   - Achieve 85% train / 80% test accuracy
   - Evaluate with classification metrics

6. **Live Prediction**
   - Load trained model and tokenizer
   - Preprocess and predict sentiment for user-typed sentences

---

## Usage

1. **Clone the repository and install dependencies**
2. **Run preprocessing scripts to generate vectorized data**
3. **Train the model using provided training scripts**
4. **Use the live prediction script to classify new tweets**

---

## Results

| Metric          | Value     |
|-----------------|-----------|
| Train Accuracy  | 85%       |
| Test Accuracy   | 80%       |
| Model           | CNN-LSTM  |
| Dataset         | [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) |

---

## Live Demo

You can input new sentences and get instant sentiment predictions using the trained model.

---

## Acknowledgements

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/)

---
