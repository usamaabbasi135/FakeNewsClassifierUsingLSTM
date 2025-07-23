# 📰 Fake News Classifier using LSTM (TensorFlow / Keras)

This project is a deep learning-based text classification model that detects fake news headlines using an LSTM (Long Short-Term Memory) neural network. The model is built using TensorFlow/Keras and trained on the **WELFake Dataset**, which contains labeled fake and real news articles.

---

## Project Overview

- **Goal:** Classify news headlines as real or fake.
- **Model:** LSTM-based binary classifier using word embeddings.
- **Data:** WELFake_Dataset.csv (pre-cleaned news titles with labels).
- **Tools & Libraries:** Python, Pandas, NLTK, TensorFlow, Scikit-learn

---

## Workflow

1. **Import Libraries**  
   Essential packages for data handling, NLP, and deep learning.

2. **Load and Explore Dataset**  
   Read CSV data, inspect shape, and clean missing values.

3. **Text Preprocessing**  
   Clean headlines using regex, remove stopwords, apply stemming with NLTK.

4. **Tokenization and Padding**  
   One-hot encode cleaned text and pad sequences for equal length.

5. **Model Building**  
   Construct a Sequential LSTM model with Embedding and Dense layers.

6. **Train/Test Split & Training**  
   Split dataset and train the model using 70/30 data split.

7. **Model Evaluation**  
   Evaluate using accuracy score, confusion matrix, and classification report.

---

## Technologies Used

- Python 3
- Pandas & NumPy
- NLTK (stopwords, stemming)
- TensorFlow / Keras
- Scikit-learn

---

## Results

- The model achieves good performance in detecting fake news headlines.
- Evaluation includes precision, recall, F1-score, and confusion matrix.

---

## Dataset Source

> Dataset: [WELFake Dataset](https://www.kaggle.com/datasets/rdaaraki/welfake-dataset)  
> This dataset combines real and fake news from multiple sources.

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/fake-news-classifier-lstm.git
   cd fake-news-classifier-lstm

## Author
- Usama Abbasi
- Data Scientis and Analyst | Afiniti
- Contact: usamahafeez.abbasi1234@gmail.com
