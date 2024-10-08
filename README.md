# Movie Review Sentiment Analysis using Streamlit

## Overview
This project demonstrates a **movie review sentiment analysis** application built using **Streamlit**. The app classifies user-provided movie reviews into **positive** or **negative** sentiment categories using a machine learning model.

## Features
- **Interactive Input**: Users can input their own movie reviews and get real-time sentiment predictions.
- **Visualization**: The app provides visual insights into sentiment distribution using charts.
- **Real-time Feedback**: Sentiment analysis is performed instantly as users input reviews.
- **Custom Review Analysis**: Upload a batch of reviews to analyze multiple inputs at once.

## How to Run the App
To run the app locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis

### 2. Install Required Libraries
Make sure you have Python installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt

### 3. Run the Streamlit App
Run the following command to launch the app:
```bash
streamlit run app.py

## Model
The sentiment analysis model used in this project is Logistic Regression. It is trained on a dataset of movie reviews and is capable of predicting positive or negative sentiments based on the input text.

