# Movie Genre Classification

## Project Overview

This project focuses on predicting the **genres of movies** based on their plot summaries using **multi-label classification**. The data used is from the **Kaggle Movie Dataset**, containing movie metadata and plot summaries. The main goal is to classify each movie into one or more genres using **Natural Language Processing (NLP)** and machine learning techniques.

## Key Features

- **Data Cleaning & Preprocessing**:
  - Removed irrelevant rows and handled missing values.
  - Applied **text cleaning** and **lemmatization** to preprocess movie plot summaries.
  - Extracted genre information from JSON format and converted it into multi-label format using **MultiLabelBinarizer**.

- **Exploratory Data Analysis (EDA)**:
  - Analyzed the frequency of different genres.
  - Visualized top genres using **bar plots** to understand their distribution in the dataset.

- **Model Training**:
  - Built a **machine learning pipeline** consisting of:
    - **Text Preprocessing**: Cleaning, lemmatization.
    - **Feature Extraction**: Using **TF-IDF Vectorizer** to convert text to numerical features.
    - **Model**: **Logistic Regression** using a **OneVsRestClassifier** to perform multi-label classification.
  - Experimented with different **thresholds** to enhance the classification accuracy.

- **Model Evaluation**:
  - Evaluated the model using **F1 Score** and **Classification Report**.
  - Predictions made for various movies, including `Titanic`, `Avatar`, and `The Conjuring`, to assess the accuracy of genre predictions.

## Dependencies

- **Programming Language**: Python 3.x
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Natural Language Processing: `nltk`, `re`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Model Persistence: `joblib`
