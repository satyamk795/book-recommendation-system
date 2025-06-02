import os
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

csv_path = os.path.join(os.path.dirname(__file__), 'data','books.csv')

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"'books.csv' not found at {csv_path}")

# Try reading with quoting, skip bad lines to avoid tokenizing errors
try:
    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines='skip')
except TypeError:
    # For older pandas versions fallback
    df = pd.read_csv(csv_path, quotechar='"', error_bad_lines=False)

print("Columns in CSV:", df.columns.tolist())

# Find description column fallback
desc_col = None
for col in ['description', 'desc', 'summary']:
    if col in df.columns:
        desc_col = col
        break

if desc_col is None:
    # create combined text for similarity
    df['desc'] = df['title'].astype(str) + " " + df['authors'].astype(str)
    desc_col = 'desc'

# Drop duplicates on title
df['title'] = df['title'].astype(str).str.strip().str.lower()
df = df.drop_duplicates(subset='title')

df = df[df[desc_col].notnull()]
df[desc_col] = df[desc_col].astype(str).str.lower()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df[desc_col])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def find_closest_title(input_title, titles, cutoff=0.6):
    matches = difflib.get_close_matches(input_title.lower(), titles, n=1, cutoff=cutoff)
    if matches:
        return matches[0]
    return None

def recommend(title):
    title = title.lower()
    closest_title = find_closest_title(title, indices.index)
    if closest_title is None:
        return []
    idx = indices[closest_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 excluding itself
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices].tolist()

def get_top_books(n=10):
    return df['title'].sample(n).tolist()

def get_book_details(titles):
    book_info = []
    for title in titles:
        row = df[df['title'] == title].iloc[0]
        book_info.append({
            'title': row['title'].title(),
            'authors': row.get('authors', 'Unknown'),
            'isbn': row.get('isbn', 'N/A')
        })
    return book_info
