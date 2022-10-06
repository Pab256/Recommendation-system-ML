import streamlit as st
import json
import recommendor as recom
import requests
from streamlit_lottie import st_lottie
st.set_page_config(page_title=' Book Recommender ', page_icon=':books:', layout='wide')

@st.cache
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie = 'https://assets10.lottiefiles.com/packages/lf20_aBYmBC.json'


def gridlayout(data, n=4):
    n_cols = n
    n_rows = 1 + len(data) // int(n_cols)
    rows = [st.container() for i in range(n_rows)]
    cols_per_row = [r.columns(n_cols) for r in rows]
    cols = [column for row in cols_per_row for column in row]
    return cols


tab1, tab2, tab3 = st.tabs(['Home', 'Recommender', 'About'])

with tab1:
    st.title('Top 50 books')
    with open('popularbooks.json', 'r') as x:
        pop = json.load(x)
    col = gridlayout(pop['image'], 4)
    for image_index, image in enumerate(pop['image']):
        col[image_index].image(image, width=150, caption=f'{pop["book"][image_index]}...'
                                                         f'Author: {pop["author"][image_index]}'
                                                         f'Rating: {str(pop["rating"][image_index])[:3]}')

with tab2:
    st.title('Searching for Books to read?')
    text = st.text_input('Enter here', help='Enter a book title you like')
    num = st.slider('Number of recommendations', min_value=5, max_value=50)
    st.button('Recommend')
    if text is not None:
        try:
            result = recom.recommend(text, num)
            col = gridlayout(result['image'], 4)
            for image_index, image in enumerate(result['image']):
                col[image_index].image(image, width=150, caption=f'{result["book"][image_index]}...'
                                                                 f'Author: {result["author"][image_index]}'
                                                                f'Rating: {str(result["rating"][image_index])[:3]}')
        except:
            st_lottie(load_lottieurl(lottie), height=300)
    else:
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
    7. os
    ''')

    st.subheader('Recommendation Engine:')
    with st.expander("recommend()"):
        st.code('''
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
        ''')

    st.subheader('Files Used:')
    with st.expander('1. Books'):
        st.dataframe(recom.books_df)
    with st.expander('2. Users'):
        st.dataframe(recom.users)
    with st.expander('3. Ratings'):
        st.dataframe(recom.ratings)

