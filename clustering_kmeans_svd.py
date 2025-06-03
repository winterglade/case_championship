import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

import nltk
import spacy

n_clusters = 18
brands = ['apple', 'bosch', 'lg', 'sony']
try:
    stop_words_en = set(nltk.corpus.stopwords.words('english'))
    stop_words_ru = set(nltk.corpus.stopwords.words('russian'))
except LookupError:
    nltk.download('stopwords')
    stop_words_en = set(nltk.corpus.stopwords.words('english'))
    stop_words_ru = set(nltk.corpus.stopwords.words('russian'))

stop_words_ru.update(['набор', 'фильтр', 'электрический'])

nlp = spacy.load("ru_core_news_sm")

def advanced_preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+[xх]\d+', '', text)  
    text = re.sub(r'\b\d+(\.\d+)?\s?(мм|см|кг|л|мл)\b', '', text)  
    text = re.sub(r'\b(?:' + '|'.join(brands) + r')\b', '', text)  
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', '', text)

    with nlp.select_pipes(disable=["ner", "parser"]):
        doc = nlp(text)
    return ' '.join([
        token.lemma_ for token in doc
        if token.lemma_ not in stop_words_ru and
           token.lemma_ not in stop_words_en and
           len(token.lemma_) >= 3 and not token.is_stop
    ])

df = pd.read_csv("train_data.csv", on_bad_lines='skip')
df['Name'] = df['Name'].fillna('')
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])

df['Processed_Name'] = df['Name'].apply(advanced_preprocess)

df['name_length'] = df['Name'].apply(len)
df['num_digits'] = df['Name'].apply(lambda x: len([c for c in x if c.isdigit()]))
df['num_words'] = df['Processed_Name'].apply(lambda x: len(x.split()))

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=5,
    max_df=0.8,
    sublinear_tf=True
)
X_text = vectorizer.fit_transform(df['Processed_Name'])
X_numeric = df[['name_length', 'num_digits', 'num_words']].values
X = hstack([X_text, X_numeric])

svd = TruncatedSVD(n_components=100, random_state=42)
X_svd = svd.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
df['KMeans_Cluster'] = kmeans.fit_predict(X_svd)

true_labels = LabelEncoder().fit_transform(df['Category'])
silhouette_avg = silhouette_score(X_svd, df['KMeans_Cluster'])
ari = adjusted_rand_score(true_labels, df['KMeans_Cluster'])

print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_svd[:, 0], y=X_svd[:, 1], hue=df['KMeans_Cluster'], palette='tab20', legend='full')
plt.title("Кластеры KMeans (SVD + TF-IDF)")
plt.savefig("results/svd_kmeans_clusters.png")
plt.show()

ct = pd.crosstab(df['KMeans_Cluster'], df['Category'])
plt.figure(figsize=(16, 10))
sns.heatmap(ct, cmap="YlGnBu", annot=True, fmt='d')
plt.title("KMeans: соответствие кластеров и категорий")
plt.ylabel("Cluster")
plt.xlabel("Category")
plt.tight_layout()
plt.savefig("results/kmeans_cluster_vs_category.png")
plt.show()
