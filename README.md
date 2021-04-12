# Short Text Topic Modeling
Unsupervised topic extraction from a body of text is one of the most frequent use cases in NLP. Commonly used methods for this task, such as [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) or [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation), rely on co-occurance of words that define different topics. This does not happen often in very short text causing those methods to fail to extract topics from short text such as article titles and social medai posts.

This repository contains code base required for building and applying two different models for topic extraction specifically designed to perform well on short text. Moreover, it includes an app to interactively apply them to arbitrary collection of text.

# Models
The first approach is **Gibbs Sampling for Drichlet Multinomial Mixture model**, or in short **GSDMM**, which is developed by [Yin & Wang 2014](dbgroup.cs.tsinghua.edu.cn). Compared to other available methods GSDMM has excellent performance while being computationally efficient (see [Qiang et al. 2019](https://arxiv.org/abs/1904.07695)). Topic modeling with this technique can be performed using the "GSDMM_model" class (see ./model/gsdmm_cls.py).

In the second approach I use transfer learning by combining sentence embedding and K-means clustering to find clusters of similar texts as proxy for the main topics. The current sentence embedding implementation uses aggregated [Glove](https://nlp.stanford.edu/projects/glove/) word embedding which I found computationally efficient while achieving [BERT](https://github.com/google-research/bert) level representation performance. Topic modeling with this technique can be done using the "Emb_Kmeans_Model" class (see ./model/emb_kmeans_cls.py).

The two aforementioned models are integrated in a web app which allows the user to upload a collection of short text (as a .csv file) and extract topics from it interactively. The user can perform topic extraction immediately using the default settings. They can also opt for choosing between the two models, adjusting text processing, modifying model parameters or even fine tune how they want to see the results. You can see the app in action in the YouTube video below:

[![Alt text](https://img.youtube.com/vi/ckn0lQPvgFw/0.jpg)](https://www.youtube.com/watch?v=ckn0lQPvgFw)

# Using Docker image
The easiest way of running the app on your local machine is to download and run its [Docker image](https://hub.docker.com/r/alistar100/alisttm).

Assuming you already have [Docker](https://www.docker.com/) downloaded installed and running, you can do this using the following command:
```
docker pull alistar100/alisttm:1.0
docker run -p 8501:8501 --detach alistar100/alisttm:1.0
```
This automatically downloads and prepare all the necessary components and you can access the app in browser through "http://localhost:8501"

# Installation using Conda
With Git and Conda already installed and up2date you need to:

1- Clone this repo:
```
git clone https://github.com/alistar/STTM-Insight.git
```

2- Make sure the submodules are also copied:

```
cd STTM-Insight
git submodule init
git submodule update
```

2- Create a conda virtual environment called 'STTM-env' using the provided .yml file. From within the STTM-Insight directory this can be done by:
```
conda env create -f configs/config.yml
conda activate STTM-env
```
# Topic extraction using the interactive App
First you need to launch the web-app locally in the "STTM-env" environment by:
```
streamlit run model/sttm_streamlit.py
```

After the last command you should be able to connect to the web-app and interact with it. The app should pop-up in your internet browser but if not, you can manually go to the given local URL streamlit indicates such as "http://localhost:8509".

You can run a demo by pressing the big bottom in the middle of the page "Extract 4 topics using Glove+K-means method". This will load a test data-set which contains 3k article titles chosen from 4 different New York Times sections, perform sentence embedding using Glove and finally finding 4 clusters using K-means clustering. The the WordCloud of the 4 clusters/topics will be shown along with their top 10 most frequent words.

_Note that when running the app for the first time with the Glove embedding method the code needs to download a file containing the Glove embeddings. The file is 274M and depending on the internet speed might take a few mins to download. This only needs to be done once and subsequent runs will be faster_

Besides using the default settings, the web-app has many features enabling the user to adjust the model and its in/outputs. It is possible to upload an arbitrary body of text as .csv file, choose the GSDMM topic extraction model instead, change the text processing steps, adjust various model parameters and the number of clusters to output.

