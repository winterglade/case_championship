import os
import re
import pickle
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


try:
    stop_words_en = set(nltk.corpus.stopwords.words('english'))
    stop_words_ru = set(nltk.corpus.stopwords.words('russian'))
except LookupError:
    nltk.download('stopwords')
    stop_words_en = set(nltk.corpus.stopwords.words('english'))
    stop_words_ru = set(nltk.corpus.stopwords.words('russian'))

stop_words_ru.update(['набор', 'фильтр', 'электрический'])
brands = ['toyota', 'ford', 'mitsubishi', 'hyundai', 'audi', 'bmw', 'mercedes', 'apple', 'bosch', 'lg', 'sony']

nlp = spacy.load("ru_core_news_sm")

def advanced_preprocess(text):
    text = text.lower()

    text = re.sub(r'\d+[xх]\d+', '', text)

    text = re.sub(r'\b\d+(\.\d+)?\s?(мм|см|кг|л|мл)\b', '', text)

    brand_pattern = r'\b(?:' + '|'.join(brands) + r')\b'
    text = re.sub(brand_pattern, '', text)

    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    with nlp.select_pipes(disable=["ner", "parser"]):
        doc = nlp(text)


    return ' '.join([
        token.lemma_ for token in doc
        if token.lemma_ not in stop_words_ru
        and token.lemma_ not in stop_words_en
        and len(token.lemma_) >= 3
        and not token.is_stop
    ])

data_path = os.path.join(os.path.dirname(__file__), "train_data.csv")
df = pd.read_csv(data_path, on_bad_lines='skip')

df['Name'] = df['Name'].fillna('')
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])

df['Processed_Name'] = df['Name'].apply(advanced_preprocess)

df[['Processed_Name']].to_csv('processed_input.csv', index=False, header=False)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 4),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    norm='l2'
)
X = vectorizer.fit_transform(df['Processed_Name'])

le = LabelEncoder()
y = le.fit_transform(df['Category'])

model = LogisticRegression(max_iter=1000, solver='liblinear')

param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X, y)

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(grid.best_estimator_, f)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Обучение завершено. Модель сохранена.")
print("Обработанный файл сохранён как 'processed_input.csv'")
print("Параметры обученной модели:")
print(grid.best_params_)
print("Тип векторизатора:", type(vectorizer))
print("Размер матрицы признаков:", X.shape)
