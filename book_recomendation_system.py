import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import streamlit as st
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title=' Book Recommender ', page_icon=':books:', layout='wide')

books = pd.read_csv("books.csv")
users = pd.read_csv("users.csv")
ratings = pd.read_csv("ratings.csv")

# popularity based recommender system
ratings_with_books = ratings.merge(books, on="ISBN")
num_rating_df = ratings_with_books.groupby("Book-Title").count()["Book-Rating"].reset_index()
num_rating_df.rename(columns={"Book-Rating": "Num_Rating"}, inplace=True)

avg_rating_df = ratings_with_books.groupby("Book-Title").mean()["Book-Rating"].reset_index()
avg_rating_df.rename(columns={"Book-Rating": "Avg_Rating"}, inplace=True)

popularity_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
popular_df = popularity_df[popularity_df["Num_Rating"] >= 250].sort_values("Avg_Rating", ascending=False).head(50)
popular_df = popular_df.merge(books, on="Book-Title").drop_duplicates("Book-Title")[
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


@st.cache
def recommend(book, no_recom=10):
    title = get_close_matches(book, pt.index, n=1, cutoff=0.5)[0]
    index = np.where(pt.index == title)[0][0]
    similar_items = dict(
        sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:no_recom])
    si = list(similar_items)
    items = []
    for i in si:
        items.append(pt.index[i])
    items.insert(0, title)
    return items


lottie2 = 'https://assets5.lottiefiles.com/packages/lf20_yg29hewu.json'
tab1, tab2, tab3 = st.tabs(['Home', 'Recommender', 'About'])

@st.cache
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()




with tab1:
    column1, column2 = st.columns(2, gap='small')
    with column2:
        st_lottie(load_lottieurl(lottie2), height=150)
    with column1:
        st.title('Top 50 books')
    image = list(popular_df['Image-URL-M'].values)
    book_name = list(popular_df['Book-Title'].values)
    author = list(popular_df['Book-Author'].values)
    rating = list(popular_df['Avg_Rating'].values)
    for i in range(len(image)):
        st.image(image[i])
        st.write('Book Name:', book_name[i])
        st.write('Author:', author[i])
        st.write('Rating:', str(rating[i])[:3])


with tab2:
    st.title('Searching for a Book to read?')
    text = st.text_input('Enter here', help='Enter a book title you like')
    num = st.slider('Number of recommendations', min_value=5, max_value=20)
    st.button('Recommend')
    if text is not None:
        try:
            result = recommend(text, num)
            image = list(books_df['Image-URL-M'].values)
            book_name = list(books_df['Book-Title'].values)
            author = list(books_df['Book-Author'].values)
            rating = list(books_df['Avg_Rating'].values)
            for j in result:
                for i in range(len(book_name)):
                    if book_name[i] == j:
                        st.image(image[i])
                        st.write('Book Name:', book_name[i])
                        st.write('Author:', author[i])
                        st.write('Rating:', str(rating[i])[:3])
        except:
            pass
with tab3:
    st.title('About the Project:')

    st.subheader('Modules used:')
    st.write('''
    1. streamlit as st
    2. numpy as np
    3. pandas as pd
    4. sklearn
    5. difflib
    6. request
    7. streamlit_lottie as st_lottie
    ''')

    st.subheader('Functions used:')
    st.write('''
    1. recommend(book_name, no_recom=10)
    2. load_lottieurl(url)''')

    st.subheader('Files Used:')
    st.write('''
    1. popular_df
    2. books_df
    ''')

    st.subheader('Recommender system:')
    st.write('Collaborative filtering')
    st.image('collab_filter_image.png')

    with st.expander("See code:"):
        st.code('''
        import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import streamlit as st
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title=' Book Recommender ', page_icon=':books:', layout='wide')

books = pd.read_csv("books.csv")
users = pd.read_csv("users.csv")
ratings = pd.read_csv("ratings.csv")

# popularity based recommender system
ratings_with_books = ratings.merge(books, on="ISBN")
num_rating_df = ratings_with_books.groupby("Book-Title").count()["Book-Rating"].reset_index()
num_rating_df.rename(columns={"Book-Rating": "Num_Rating"}, inplace=True)

avg_rating_df = ratings_with_books.groupby("Book-Title").mean()["Book-Rating"].reset_index()
avg_rating_df.rename(columns={"Book-Rating": "Avg_Rating"}, inplace=True)

popularity_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
popular_df = popularity_df[popularity_df["Num_Rating"] >= 250].sort_values("Avg_Rating", ascending=False).head(50)
popular_df = popular_df.merge(books, on="Book-Title").drop_duplicates("Book-Title")[
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


@st.cache
def recommend(book, no_recom=10):
    title = get_close_matches(book, pt.index, n=1, cutoff=0.5)[0]
    index = np.where(pt.index == title)[0][0]
    similar_items = dict(
        sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:no_recom])
    si = list(similar_items)
    items = []
    for i in si:
        items.append(pt.index[i])
    items.insert(0, title)
    return items


lottie2 = 'https://assets5.lottiefiles.com/packages/lf20_yg29hewu.json'
tab1, tab2, tab3 = st.tabs(['Home', 'Recommender', 'About'])

@st.cache
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()




with tab1:
    column1, column2 = st.columns(2, gap='small')
    with column2:
        st_lottie(load_lottieurl(lottie2), height=150)
    with column1:
        st.title('Top 50 books')
    image = list(popular_df['Image-URL-M'].values)
    book_name = list(popular_df['Book-Title'].values)
    author = list(popular_df['Book-Author'].values)
    rating = list(popular_df['Avg_Rating'].values)
    for i in range(len(image)):
        st.image(image[i])
        st.write('Book Name:', book_name[i])
        st.write('Author:', author[i])
        st.write('Rating:', str(rating[i])[:3])


with tab2:
    st.title('Searching for a Book to read?')
    text = st.text_input('Enter here', help='Enter a book title you like')
    num = st.slider('Number of recommendations', min_value=5, max_value=20)
    st.button('Recommend')
    if text is not None:
        try:
            with st.spinner('In Progress..'):
                result = recommend(text, num)
                image = list(books_df['Image-URL-M'].values)
                book_name = list(books_df['Book-Title'].values)
                author = list(books_df['Book-Author'].values)
                rating = list(books_df['Avg_Rating'].values)
                for j in result:
                    for i in range(len(book_name)):
                        if book_name[i] == j:
                            st.image(image[i])
                            st.write('Book Name:', book_name[i])
                            st.write('Author:', author[i])
                            st.write('Rating:', str(rating[i])[:3])
        except:
            pass
with tab3:
    st.title('About the Project:')

    st.subheader('Modules used:')
    st.write(...)

    st.subheader('Functions used:')
    st.write(...)

    st.subheader('Files Used:')
    st.write(...)

    st.subheader('Recommender system:')
    st.write('Collaborative filtering')
    st.image('collab_filter_image.png')

    with st.expander("See code:"):
        st.code(...)

        ''')
