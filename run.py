import sys
import io
import pandas as pd
import pickle
import argparse
import os
import spacy
import nltk
from nltk.corpus import stopwords

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    stop_words_en = set(stopwords.words('english'))
    stop_words_ru = set(stopwords.words('russian'))
except LookupError:
    print("Скачивание стоп-слов для NLTK...")
    nltk.download('stopwords') 
    stop_words_en = set(stopwords.words('english'))
    stop_words_ru = set(stopwords.words('russian'))

stop_words_ru.update(['набор', 'фильтр', 'электрический'])

try:
    nlp = spacy.load("ru_core_news_sm")
except OSError:
    from spacy.cli import download
    download("ru_core_news_sm")
    nlp = spacy.load("ru_core_news_sm")

def preprocess_text(text):
    text = text.lower()
    with nlp.select_pipes(disable=["ner", "parser"]):
        doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc
                     if token.lemma_ not in stop_words_ru and
                        token.lemma_ not in stop_words_en and
                        len(token.lemma_) >= 3])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', type=str, required=True,
                        help='Path to input .csv (no header)')
    parser.add_argument('--output-path', '-o', type=str, required=True,
                        help='Path to save predictions')
    args = parser.parse_args()

    with open(args.input_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    X_test = [preprocess_text(line.strip()) if line.strip() != '' else 'unknown' for line in lines]

    try:
        with open("models/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        if not hasattr(vectorizer, 'idf_'):
            print("Векторизатор не обучен. Обучите векторизатор на тренировочных данных.")
            raise ValueError("Векторизатор не обучен.")
        print("Векторизатор успешно загружен!")
    except Exception as e:
        print(f"Ошибка загрузки векторизатора: {e}")
        raise

    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        raise

    try:
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        print("Label encoder успешно загружен!")
    except Exception as e:
        print(f"Ошибка загрузки label_encoder: {e}")
        raise

    try:
        X_test_tfidf = vectorizer.transform(X_test)
        print(f"TF-IDF преобразование успешно завершено. Размерность: {X_test_tfidf.shape}")
    except Exception as e:
        print(f"Ошибка преобразования текста в TF-IDF: {e}")
        raise

    try:
        y_pred = model.predict(X_test_tfidf)
        labels = label_encoder.inverse_transform(y_pred)
        print(f"Предсказания успешно выполнены для {len(labels)} строк.")
    except Exception as e:
        print(f"Ошибка при выполнении предсказания: {e}")
        raise

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        pd.Series(labels).to_csv(args.output_path, index=False, header=False, encoding='utf-8')
        print(f"Предсказания сохранены в {args.output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        raise

    print(pd.Series(labels).head())
