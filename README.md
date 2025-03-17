# Natural Language Processing (NLP) Project

## 📌 Overview
This project focuses on **Natural Language Processing (NLP)** using a dataset of Yelp reviews. The goal is to analyze text data, preprocess it, and build machine learning models to classify reviews based on sentiment.

## 📂 Project Structure
```
├── data/                  # Dataset used for analysis (yelp.csv)
├── notebooks/             # Jupyter notebooks with step-by-step implementations
├── src/                   # Python scripts for text processing and model training
├── results/               # Outputs, evaluation metrics, and reports
├── visualizations/        # Plots and charts generated during EDA
├── README.md              # Project documentation
└── requirements.txt       # Dependencies required to run the project
```

## 📊 Dataset
The dataset used in this project is **Yelp reviews**, containing 10,000 reviews with the following features:
- **business_id**: Unique identifier for businesses.
- **date**: Date of the review.
- **review_id**: Unique identifier for each review.
- **stars**: Ratings given by users (1 to 5 stars).
- **text**: The actual review content.
- **type**: Data type (all entries are "review").
- **user_id**: Unique identifier for the user.
- **cool, useful, funny**: Vote counts for each category.

## 🚀 Installation
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/27abhishek27/NLP-Natural-Language-Processing-Project.git
cd NLP-Natural-Language-Processing-Project
```

### 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔍 Methodology
### 1. **Data Preprocessing**
- **Text Cleaning**: Removed special characters, stopwords, and performed tokenization.
- **Feature Engineering**: Extracted word frequencies and TF-IDF features.
- **Label Encoding**: Converted star ratings into categorical sentiment labels.

### 2. **Exploratory Data Analysis (EDA)**
- **Word Cloud**: Identified most common words in reviews.
- **Sentiment Distribution**: Visualized positive and negative review patterns.
- **N-grams Analysis**: Explored frequent word sequences.

### 3. **Model Training**
- **Naive Bayes Classifier**: Applied for sentiment classification.
- **Logistic Regression**: Used for binary classification.
- **Support Vector Machine (SVM)**: Implemented for improved accuracy.
- **Hyperparameter Tuning**: Used `GridSearchCV` for optimization.

### 4. **Model Evaluation**
- **Accuracy, Precision, Recall, F1-score**: Evaluated model performance.
- **Confusion Matrix**: Analyzed false positives and false negatives.
- **ROC Curve**: Assessed classification confidence.

## 📊 Visualizations
All generated plots and graphs are stored in the `NLP Project png/` folder. These include:
- **FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings**: ![5 Histogram](https://github.com/27abhishek27/NLP-Natural-Language-Processing-Project/blob/main/NLP%20project%20png/5%20histogram.png)
- **Boxplot of text length for each star category**: ![Boxplot](https://github.com/27abhishek27/NLP-Natural-Language-Processing-Project/blob/main/NLP%20project%20png/boxplot%20of%20text%20length%20for%20each%20star%20category.png)
- **Countplot of the number of occurrences for each type of star rating**: ![Countplot](https://github.com/27abhishek27/NLP-Natural-Language-Processing-Project/blob/main/NLP%20project%20png/countplot%20of%20the%20number%20of%20occurrences%20for%20each%20type%20of%20star%20rating.png)
- **Seaborn to create a heatmap based off that .corr() dataframe**: ![Heatmap](https://github.com/27abhishek27/NLP-Natural-Language-Processing-Project/blob/main/NLP%20project%20png/%20seaborn%20to%20create%20a%20heatmap%20based%20off%20that%20.corr()%20dataframe.png)

## 🛠️ Technologies Used
- **Python**
- **NLTK & SpaCy**
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Jupyter Notebook**

## 📌 Future Improvements
- **Deep Learning Models**: Implement LSTMs and transformers.
- **Multi-class Sentiment Analysis**: Improve classification beyond positive/negative.
- **Real-time Sentiment Analysis**: Deploy model as an API.


