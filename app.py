from flask import Flask, render_template, request
from model import recommend, get_top_books, get_book_details, find_closest_title, indices

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recs = []
    error = ""
    matched_title = None
    top_books = get_book_details(get_top_books())

    if request.method == 'POST':
        user_input = request.form.get('book', '').strip()
        if user_input:
            closest = find_closest_title(user_input, indices.index)
            if closest is None:
                error = "Sorry, book not found. Try another one."
            else:
                matched_title = closest.title()
                rec_titles = recommend(closest)
                recs = get_book_details(rec_titles)
        else:
            error = "Please enter a book title."

    return render_template("index.html", recs=recs, error=error, top_books=top_books, matched_title=matched_title)

if __name__ == "__main__":
    app.run(debug=True)
