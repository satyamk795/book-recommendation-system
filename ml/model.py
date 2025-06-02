import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv("ml/books.csv")

df = df.drop_duplicates(subset='title')
df = df[df['desc'].notnull()]
df['desc'] = df['desc'].fillna('').str.lower()
df['title'] = df['title'].str.strip().str.lower()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['desc'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title):
    title = title.lower()
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices].tolist()

def get_top_books(n=10):
    return df['title'].sample(n).tolist()

def get_book_details(titles):
    book_info = []
    for title in titles:
        row = df[df['title'] == title].iloc[0]
        isbn = row['isbn']
        book_info.append({
            'title': title,
            'isbn': isbn
        })
    return book_info
