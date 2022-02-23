import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from nltk.stem.snowball import SnowballStemmer
from train import stemming, find_top_predict

st.set_page_config(page_title=None, page_icon=None, layout='centered', initial_sidebar_state='collapsed')

"""
# Econ JEL Code Match

"""


@st.cache(show_spinner=False, allow_output_mutation=True)
def load_file_cached():
    jel_des = pd.read_csv("Data/jel_des.csv", index_col="jel")
    jel_labels = joblib.load("Data/jel_labels.pkl")
    vectorizer, classifier = joblib.load("Data/model.pkl")
    return jel_des, jel_labels, vectorizer, classifier


# def load_file():
#     update_timestamp = os.path.getmtime("Data/model.pkl")
#     return load_file_cached(update_timestamp)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def hide_right_menu():
    # ref: https://discuss.streamlit.io/t/how-do-i-hide-remove-the-menu-in-production/362/3
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def predict(abstract, vectorizer, classifier, jel_labels, top_n):
    stemmer = SnowballStemmer("english")
    abstract = stemming(abstract, stemmer)
    X = vectorizer.transform([abstract])
    y_predict_jel, _ = find_top_predict(classifier, X, jel_labels, top_n=top_n)
    return y_predict_jel


def show_jels(y_predict_jel, jel_des):
    # for jel in y_predict_jel:
    #     if jel in jel_des.index.values:
    #         des = jel_des.loc[jel]
    #         st.markdown(f'{jel}: {des.des} - {des.d2c}: {des.d2d}')

    # @ grouped show
    jels = [jel for jel in y_predict_jel if jel in jel_des.index.values]
    dess = jel_des.loc[jels]
    for d2c, dess_group in dess.groupby("d2c", sort=False):
        st.markdown(f"{d2c}: {dess_group.d2d.unique()[0]}")
        for d3c, row in dess_group.iterrows():
            f" - {d3c}: {row['des']}"


def sidebar_info():
    st.sidebar.header("About")
    st.sidebar.markdown("""
    <div style="font-size: small;">
    This is a simple app to match JEL codes for economics papers based on the abstract text.<br>
    It uses a simple logistic regression model trained on real economics publications in last 20 years.<br>
    A few JEL codes that are rare in the literature are dropped, and <font style="color:green;">the model favors the JEL codes that are common</font>.<br>
    The results shown are the top 20 JEL codes predicted by the model and you can choose to show more or less by setting the config below.<br>
    The model currectly has <font style="color:green;">over 70 percent accuracy (true positive)</font> on the test sample.<br>
    If you feel that the results are not satisfactory, you can try <font style="color:green;">adding more representative texts from your paper or directly entering the keywords that you are looking for (highly recommended).</font><br>
    </div>
    """, unsafe_allow_html=True)  # Author: Xuanli Zhu.<br>

    st.sidebar.header("Config")
    top_n = st.sidebar.slider('# of top JEL codes shown', value=20, min_value=10, max_value=30)

    st.sidebar.header("Report Issues")
    st.sidebar.markdown("""
    <div style="font-size: small">
    Report an issue or a suggestion at <a href="https://github.com/Alalalalaki/Econ-JEL-Match" target="_blank" rel="noopener noreferrer">github repo</a>
    </div>
    """, unsafe_allow_html=True)
    return top_n


def main():
    top_n = sidebar_info()
    hide_right_menu()
    local_css("Code/style.css")

    form = st.form(key='search')

    abstract = form.text_area('Enter Abstract', height=300)
    a1, a2 = form.columns([3.9, 1])
    button_clicked = a1.form_submit_button(label='Match')
    a2.markdown(
        """<div style="color: green; font-size: small;">
        (see left sidebar for help)
        </div>""",
        unsafe_allow_html=True)

    jel_des, jel_labels, vectorizer, classifier = load_file_cached()

    jel_load_state = st.empty()

    if button_clicked:
        jel_load_state.markdown('Matching ...')
        y_predict_jel = predict(abstract, vectorizer, classifier, jel_labels, top_n)
        jel_load_state.markdown(f'Top {top_n} JEL Codes Matched:')
        show_jels(y_predict_jel, jel_des)


if __name__ == '__main__':
    main()
