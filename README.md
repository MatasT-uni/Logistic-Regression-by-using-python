---

### README

# Tweet Sentiment Analysis using Logistic Regression

This project focuses on implementing a Logistic Regression model from scratch to perform sentiment analysis on tweets. The goal is to classify tweets into positive or negative sentiment categories. The project walks through the steps of feature extraction, model training, testing, and error analysis to evaluate the performance of the Logistic Regression model.

## Project Overview

This project is structured as an assignment to understand the core concepts of Logistic Regression in the context of natural language processing (NLP). The task is to predict whether a given tweet has a positive or negative sentiment based on the features extracted from the text.

### Key Objectives

- **Feature Extraction**: Extract features from tweets to use as input for the Logistic Regression model.
- **Model Implementation**: Implement Logistic Regression from scratch, including the sigmoid function, cost function, gradient descent, and prediction functions.
- **Model Training**: Train the Logistic Regression model using the extracted features.
- **Error Analysis**: Analyze the errors by reviewing misclassified tweets to understand where the model might be making mistakes.
- **Custom Predictions**: Allow users to input their own tweets and see the sentiment prediction.

## Notebook Structure

1. **Part 1: Logistic Regression**:
   - **Sigmoid Function**: Implement the sigmoid function used for classification.
   - **Cost Function and Gradient**: Define the cost function and gradients used for optimizing the model.
   - **Gradient Descent**: Implement the gradient descent function to minimize the cost and learn the model parameters.

2. **Part 2: Extracting the Features**:
   - Extract features such as the number of positive and negative words in a tweet to use as inputs for the model.

3. **Part 3: Training Your Model**:
   - Train the Logistic Regression model using the features extracted from the tweets.

4. **Part 4: Test Your Logistic Regression**:
   - Test the model on a validation set to evaluate its performance in predicting tweet sentiments.

5. **Part 5: Error Analysis**:
   - Analyze tweets that were misclassified by the model to understand potential reasons for errors.

6. **Part 6: Predict with Your Own Tweet**:
   - Input your own tweet and use the trained model to predict whether the sentiment is positive or negative.

## Getting Started

### Prerequisites

To run this project, you will need:

- **Python 3.x**: The programming language used for this project.
- **Jupyter Notebook**: To run the `.ipynb` file.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MatasT-uni/Logistic-Regression-by-using-python
   cd Logistic-Regression-by-using-python
   ```

2. **Install Required Packages**:
   Make sure you have the necessary Python packages installed. If not, you can install them using:
   ```bash
   pip install numpy pandas matplotlib
   pip install nltk
   ```

3. **Run the Notebook**:
   Open the Jupyter Notebook and run all cells to see the implementation of the Logistic Regression model and its application on tweet sentiment analysis.

### Usage

- **Run the notebook** to see the step-by-step implementation of Logistic Regression.
- **Use the trained model** to predict the sentiment of new tweets by modifying the input in the relevant cells.
- **Analyze errors** by reviewing misclassified examples to gain insights into potential model improvements.

## Results

The project demonstrates the process of building a Logistic Regression model from scratch and applying it to a real-world problem of sentiment analysis. Through this process, the model's accuracy and performance can be evaluated, and insights into model behavior can be gained through error analysis.

---
