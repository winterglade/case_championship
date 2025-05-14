import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import re
import nltk
import spacy

try:
    stop_words_en = set(nltk.corpus.stopwords.words('english'))
    stop_words_ru = set(nltk.corpus.stopwords.words('russian'))
except LookupError:
    nltk.download('stopwords')
    stop_words_en = set(nltk.corpus.stopwords.words('english'))
    stop_words_ru = set(nltk.corpus.stopwords.words('russian'))

nlp = spacy.load("ru_core_news_sm")

visualization_dir = "visualizations"
os.makedirs(visualization_dir, exist_ok=True)

df = pd.read_csv("train_data.csv")
df['Name'] = df['Name'].fillna('')
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', '', text)
    return ' '.join([
        token.lemma_ for token in nlp(text)
        if token.lemma_ not in stop_words_ru and
           token.lemma_ not in stop_words_en and
           len(token.lemma_) >= 3 and not token.is_stop
    ])

df['Processed_Name'] = df['Name'].apply(preprocess_text)

category_counts = df['Category'].value_counts()

def plot_category_distribution(data, category_counts):
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=data,
        x='Category',
        order=category_counts.index,
        palette='Set2',
        hue='Category',
        legend=False
    )
    plt.xticks(rotation=90)
    plt.title('Распределение категорий товаров')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(visualization_dir, 'category_distribution.png'))
    plt.close()

def plot_name_length_distribution(data):
    data['name_length'] = data['Name'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['name_length'], bins=30, kde=True, color='blue')
    plt.title('Распределение длины названий товаров')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(visualization_dir, 'name_length_distribution.png'))
    plt.close()

def plot_top_words_per_category(data, top_categories):
    for category in top_categories:
        category_data = data[data['Category'] == category]
        category_data.loc[:, 'Processed_Name'] = category_data['Processed_Name'].fillna('')
        all_words = ' '.join(category_data['Processed_Name'].astype(str)).split()
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(10)

        if not common_words:
            continue

        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts)
        plt.title(f'Топ 10 слов для категории: {category}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(visualization_dir, f'top_words_category_{category}.png'))
        plt.close()

def plot_word_cloud(text, title, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(visualization_dir, filename))
    plt.close()

def plot_top_ngrams(corpus, n=2, top_k=10, title='Top n-grams', filename='top_ngrams.png'):
    vectorizer = CountVectorizer(ngram_range=(n, n), max_features=5000)
    X = vectorizer.fit_transform(corpus)
    sum_words = X.sum(axis=0)
    word_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_ngrams = sorted(word_freq, key=lambda x: x[1], reverse=True)[:top_k]

    if not sorted_ngrams:
        return

    words, counts = zip(*sorted_ngrams)
    ngram_df = pd.DataFrame({'ngram': words, 'count': counts})

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=ngram_df,
        y='ngram',
        x='count',
        hue='ngram',
        dodge=False,
        palette=sns.color_palette("viridis", len(words)),
        legend=False
    )
    plt.title(title)
    plt.xlabel('Частота')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(visualization_dir, filename))
    plt.close()

def main():
    plot_category_distribution(df, category_counts)

    plot_name_length_distribution(df)

    top_categories = category_counts.head(5).index
    plot_top_words_per_category(df, top_categories)

    all_words_text = ' '.join(df['Processed_Name'].astype(str))
    plot_word_cloud(all_words_text, "Word Cloud по всем товарам", 'word_cloud_all.png')

    for category in top_categories:
        category_data = df[df['Category'] == category]
        all_words = ' '.join(category_data['Processed_Name'].astype(str))

        plot_word_cloud(all_words, f"Word Cloud: {category}", f'word_cloud_category_{category}.png')

        plot_top_ngrams(
            corpus=category_data['Processed_Name'].astype(str),
            n=2,
            top_k=10,
            title=f"Топ биграмм: {category}",
            filename=f'bigrams_{category}.png'
        )

        plot_top_ngrams(
            corpus=category_data['Processed_Name'].astype(str),
            n=3,
            top_k=10,
            title=f"Топ триграмм: {category}",
            filename=f'trigrams_{category}.png'
        )

    plot_top_ngrams(
        corpus=df['Processed_Name'].astype(str),
        n=2,
        top_k=15,
        title="Топ 15 биграмм по всем товарам",
        filename='top_bigrams_all.png'
    )

    plot_top_ngrams(
        corpus=df['Processed_Name'].astype(str),
        n=3,
        top_k=15,
        title="Топ 15 триграмм по всем товарам",
        filename='top_trigrams_all.png'
    )

if __name__ == "__main__":
    main()
