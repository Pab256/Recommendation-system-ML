from difflib import get_close_matches
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

books = pd.read_csv("books.csv")
users = pd.read_csv("users.csv")
ratings = pd.read_csv("ratings.csv")
# preprocessing

# popularity based recommender system
ratings_with_books = ratings.merge(books, on="ISBN")
num_rating_df = ratings_with_books.groupby("Book-Title").count()["Book-Rating"].reset_index()
num_rating_df.rename(columns={"Book-Rating": "Num_Rating"}, inplace=True)

avg_rating_df = ratings_with_books.groupby("Book-Title").mean()["Book-Rating"].reset_index()
avg_rating_df.rename(columns={"Book-Rating": "Avg_Rating"}, inplace=True)

popularity_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
data = popularity_df[popularity_df["Num_Rating"] >= 250].sort_values("Avg_Rating", ascending=False).head(50)
data = data.merge(books, on="Book-Title").drop_duplicates("Book-Title")[
    ["Book-Title", "Book-Author", "Image-URL-M", "Num_Rating", "Avg_Rating"]]

books_df = popularity_df.sort_values("Avg_Rating", ascending=False)
books_df = books_df.merge(books, on="Book-Title").drop_duplicates("Book-Title")[
    ["Book-Title", "Book-Author", "Image-URL-M", "Num_Rating", "Avg_Rating"]]

# Collaborative Filtering Based Recommender System
x = ratings_with_books.groupby("User-ID").count()["Book-Rating"] > 200
users_who_vote = x[x].index
filtered_rating = ratings_with_books[ratings_with_books["User-ID"].isin(users_who_vote)]
y = filtered_rating.groupby("Book-Title").count()["Book-Rating"] >= 50
Books_read_bymany = y[y].index
final_ratings = filtered_rating[filtered_rating["Book-Title"].isin(Books_read_bymany)]
pt = final_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
pt.fillna(0, inplace=True)

similarity_scores = cosine_similarity(pt)


def make_dictionary(data):
    dic = {'image':list(data['Image-URL-M'].values),
           'book':list(data['Book-Title'].values),
           'author':list(data['Book-Author'].values),
           'rating':list(data['Avg_Rating'].values)}
    return dic
def doesFileExists(filePathAndName):
    return os.path.exists(filePathAndName)

if doesFileExists('./books.json'):
    pass
else:
    json.dump(make_dictionary(books_df), 'books.json')
    json.dump(make_dictionary(data), 'popularbooks.json')

def recommend(book, no_recom=5):
    with open('books.json', 'r') as x:
        books_details = json.load(x)
    title = get_close_matches(book, pt.index, n=1, cutoff=0.5)[0]
    index = np.where(pt.index == title)[0][0]
    similar_items = dict(
        sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:no_recom])
    si = list(similar_items)
    items = [title]+[pt.index[i] for i in si]
    details = {"book":[], "author":[], "image":[], "rating":[]}
    for i in items:
        index=books_details['book'].index(i)
        details['book']+=[books_details['book'][index]]
        details['author']+=[books_details['author'][index]]
        details['image']+=[books_details['image'][index]]
        details['rating']+=[books_details['rating'][index]]
    return details

