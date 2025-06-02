from flask import Flask, render_template, request
from ml.model import recommend, get_top_books, get_book_details

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recs = []
    error = ""
    top_books = get_book_details(get_top_books())

    if request.method == 'POST':
        title = request.form.get('book').strip()
        rec_titles = recommend(title)
        if not rec_titles:
            error = "Sorry, book not found. Try another one."
        else:
            recs = get_book_details(rec_titles)

    return render_template("index.html", recs=recs, error=error, top_books=top_books)

if __name__ == "__main__":
    app.run(debug=True)
