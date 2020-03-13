import pickle
import random
import sys
import os
import time

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from google_drive_downloader import GoogleDriveDownloader as gdd

sys.path.append(os.getcwd())

from model.emb_kmeans_cls import Emb_Kmeans_Model
from model.gsdmm_cls import GSDMM_Model
from processing.text_processing import *

matplotlib.use("Agg")


@st.cache
def load_emb():
    try:
        with open("./Glove_quora_words.pkl", "rb") as file2open:
            embedding_cls = pickle.load(file2open)
    except:
        file_id = "1AgWnGC5pwS96u5zBjk_LfkAdDAuSkLcC"
        dest_path = "./Glove_quora_words.pkl"
        gdd.download_file_from_google_drive(
            file_id=file_id, dest_path=dest_path, unzip=True
        )
        with open("./Glove_quora_words.pkl", "rb") as file2open:
            embedding_cls = pickle.load(file2open)
    emb_matrix = embedding_cls.matrix()
    word2index = embedding_cls.word2index
    return word2index, emb_matrix


def load_data():
    try:
        df_data = pd.read_csv("../data/NYT_4topics_int.csv")
    except:
        file_id = "1BBy_zvlD6RunymaxcXGCSEw2At2KpqHk"
        dest_path = "./NYT_4topics_int.csv"
        gdd.download_file_from_google_drive(
            file_id=file_id, dest_path=dest_path, unzip=True
        )
        df_data = pd.read_csv("./NYT_4topics_int.csv")
    column2read = "text"
    return df_data[column2read].to_list()


@st.cache
def setup_model_Kmeans(
    text4topics=None,
    lowercase=True,
    remove_punct=True,
    strip_non_ascii=True,
    remove_digits=True,
    remove_stopwords=True,
    lemmatize=True,
    n_class=4,
    max_iter=1000,
    max_no_improvement=50,
    random_state=4,
    reassignment_ratio=0.0001,
    batch_size=100,
    tol=1e-05,
):
    word2index, emb_matrix = load_emb()
    model = Emb_Kmeans_Model(
        corpus=text4topics,
        word2index=word2index,
        emb_matrix=emb_matrix,
        agg_func="mean",
        lowercase=lowercase,
        remove_punct=remove_punct,
        strip_non_ascii=strip_non_ascii,
        remove_digits=remove_digits,
        remove_stopwords=remove_stopwords,
        lemmatize=lemmatize,
        n_class=n_class,
        max_iter=max_iter,
        max_no_improvement=max_no_improvement,
        random_state=random_state,
        reassignment_ratio=reassignment_ratio,
        tol=tol,
        batch_size=batch_size,
        verbose=False,
    )
    return model


@st.cache
def setup_model_GSDMM(
    text4topics=None,
    lowercase=True,
    remove_punct=True,
    strip_non_ascii=True,
    remove_digits=True,
    remove_stopwords=True,
    lemmatize=True,
    n_class=4,
    alpha=0.7,
    beta=0.6,
    n_iter=30,
):

    model = GSDMM_Model(
        corpus=text4topics,
        lowercase=lowercase,
        remove_punct=remove_punct,
        strip_non_ascii=strip_non_ascii,
        remove_digits=remove_digits,
        remove_stopwords=remove_stopwords,
        lemmatize=lemmatize,
        n_class=n_class,
        alpha=alpha,
        beta=beta,
        n_iter=n_iter,
    )
    return model


# @st.cache
def show_insight(
    n_doc_per_c,
    fractions,
    populars,
    freq_dists,
    show_wordclouds=True,
    nc2show=4,
    nword2show=20,
):
    if show_wordclouds:
        wordcloud = WordCloud(
            background_color="white",
            min_font_size=4,
            max_words=30,
            prefer_horizontal=0.7,
            contour_color="steelblue",
        )

    try:
        for i in range(nc2show):
            st.subheader(
                f"Cluster {i+1} contains {n_doc_per_c[i]} pieces (%{fractions[i]:.0f})"
            )
            st.subheader(f"Top words:")
            str_words_freq = ""
            j = 0
            this_cluster_words = freq_dists[i]
            if len(this_cluster_words) < 1:
                st.error(f"No words in this cluster!")
                break
            for key, value in this_cluster_words.items():
                if j >= nword2show:
                    break
                j += 1
                str_words_freq += f"\t{j}- {key} ({value}) "
            st.write(f"{str_words_freq}")
            if show_wordclouds and len(this_cluster_words) > 0:
                plt.imshow(
                    wordcloud.generate_from_frequencies(frequencies=this_cluster_words)
                )
                plt.axis("off")
                st.pyplot()

    except IndexError:
        st.error(f"No more clusters to show!")
    return


if __name__ == "__main__":

    text4topics = load_data()
    if st.sidebar.checkbox("Upload a file"):
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df_data = pd.read_csv(uploaded_file)
            if st.sidebar.checkbox("Sample the input"):
                sample_size = st.sidebar.slider(
                    "Sample size",
                    min_value=1,
                    max_value=df_data.shape[0],
                    value=int(df_data.shape[0] * 0.5),
                    step=500,
                    format="%i",
                )
                if sample_size > df_data.shape[0]:
                    sample_size = df_data.shape[0]
                    st.sidebar.warning(
                        f"sample size set to data length of {df_data.shape[0]}"
                    )
                df_data = df_data.sample(sample_size)
            if st.sidebar.checkbox("Have a look at data"):
                st.sidebar.dataframe(df_data.head(10))
            columns = list(df_data.columns)
            column2read = st.sidebar.selectbox(
                "Which columnd contain the text?", columns
            )

            text4topics = list(df_data[column2read])
    else:
        st.write(
            "<font color=#000080>Using a sample of 3k titles from 4 different NYT article categories. </br> Upload a file using sidebar to replace it.</font>",
            unsafe_allow_html=True,
        )
    if st.sidebar.checkbox("Tune text processing"):
        lowercase = st.sidebar.checkbox("LowerCase")
        remove_punct = st.sidebar.checkbox("Remove Punctuations")
        strip_non_ascii = st.sidebar.checkbox("Remove Non-Ascii")
        remove_digits = st.sidebar.checkbox("Remove Digits")
        remove_stopwords = st.sidebar.checkbox("Remove Stopwords")
        # lemmatize = st.sidebar.checkbox("Lemmatize")
    else:
        lowercase = True
        remove_punct = True
        strip_non_ascii = True
        remove_digits = True
        remove_stopwords = True
        lemmatize = False  # True

    model_choice = st.sidebar.radio("Which model?", ("Glove+K-means", "GSDMM"))

    n_class = st.sidebar.number_input(
        "How many classes", format="%i", min_value=2, max_value=50, value=4, step=1
    )
    if st.sidebar.checkbox("Tune the model"):
        if model_choice == "Glove+K-means":
            max_iter = st.sidebar.slider(
                "Max # of iterations",
                min_value=1,
                max_value=5000,
                value=1000,
                step=500,
                format="%i",
            )
            batch_size = st.sidebar.slider(
                "Batch size",
                min_value=1,
                max_value=1000,
                value=200,
                step=100,
                format="%i",
            )
        else:
            n_iter = st.sidebar.slider(
                "# of iterations",
                min_value=1,
                max_value=100,
                value=10,
                step=10,
                format="%i",
            )
            alpha = st.sidebar.slider(
                "alpha", min_value=0.0, max_value=1.0, value=0.1, step=0.1, format="%f"
            )
            beta = st.sidebar.slider(
                "beta", min_value=0.0, max_value=1.0, value=0.1, step=0.1, format="%f"
            )
    else:
        if model_choice == "Glove+K-means":
            max_iter = 1000
            batch_size = n_class * 25
        else:
            n_iter = 10
            alpha = 0.1
            beta = 0.1

    def_nc2show = int(n_class)
    if st.sidebar.checkbox("Fine-tune the outputs"):
        nc2show = st.sidebar.slider(
            "How many clusters to show?",
            min_value=1,
            max_value=n_class,
            value=def_nc2show,
            step=1,
            format="%i",
        )
        nword2show = st.sidebar.slider(
            "How many words per cluster?",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            format="%i",
        )
    else:
        nc2show = def_nc2show
        nword2show = 10

    show_wordclouds = st.sidebar.checkbox("Show WordClouds", value=True)

    model = None
    model_is_loaded = False
    if st.button(f"Extract {n_class} topics using {model_choice} method"):
        model_is_loaded = True
        if model_choice == "Glove+K-means":
            model = setup_model_Kmeans(
                text4topics=text4topics,
                lowercase=lowercase,
                remove_punct=remove_punct,
                strip_non_ascii=strip_non_ascii,
                remove_digits=remove_digits,
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize,
                n_class=n_class,
                max_iter=max_iter,
                batch_size=batch_size,
            )
        else:
            model = setup_model_GSDMM(
                text4topics=text4topics,
                lowercase=lowercase,
                remove_punct=remove_punct,
                strip_non_ascii=strip_non_ascii,
                remove_digits=remove_digits,
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize,
                n_class=n_class,
                alpha=alpha,
                beta=beta,
                n_iter=n_iter,
            )

    if model_is_loaded:
        n_doc_per_c, fractions, populars, freq_dists = model.inferences()

        show_insight(
            n_doc_per_c=n_doc_per_c,
            fractions=fractions,
            populars=populars,
            freq_dists=freq_dists,
            show_wordclouds=show_wordclouds,
            nc2show=nc2show,
            nword2show=nword2show,
        )
