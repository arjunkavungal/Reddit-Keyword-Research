import pickle
from flask import send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import praw
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

pickle.dump(Pipeline([('vectorizer', TfidfVectorizer()), ('classifier',LinearRegression())]), open('model.pkl', 'wb'))

reddit = praw.Reddit(client_id="p1lt136fs51SWOv6zlM6QA",client_secret="Ffth4WUUPhmFO4_b6oUdlmY5e3ZOYA",
                     username="Hot-Helicopter5986",password="",user_agent="a")
def get_hot_titles(keyword):
    subreddit = reddit.subreddit(keyword)
    hot = subreddit.hot(limit=20)
    df = pd.DataFrame()
    for i in hot:
        df = df.append({'title': i.title,'ups':i.ups}, ignore_index=True)
    return df

def get_title_length(df):
    df['Title length'] = len(df['title'])
    for i in range(len(df)):
        df['Title length'][i] = len(df['title'][i])
    return df
def title_length_graph(df):
    q_low = df["Title length"].quantile(0.01)
    q_hi  = df["Title length"].quantile(0.99)

    df_filtered = df[(df["Title length"] < q_hi) & (df["Title length"] > q_low)]
    sns.set(style='whitegrid')
    
    fig,ax=plt.subplots(figsize=(6,6))
    ax=sns.set(style="darkgrid")
    sns.scatterplot(df['Title length'],df['ups'])
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return img

def get_word_count(df):
    a = []
    for i in range(len(df)):
        a.append(len(df['title'][i].split(' ')))
    df['Word count'] = a 
    return df
def graph_scatter_plot(df, x, y, xlabel, ylabel, title):
    #fig = plt.figure(figsize=(10, 4))
    plt.scatter(df[x],df[y])
    plt.xlabel("Word count")
    plt.ylabel("Number of Upvotes")
    plt.title("Effect of Word Count on Number of Upvotes")
def unweighted_word_count(df):
    s = ""
    for i in range(len(df)):
        s += df['title'][i] + " "
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in s:
        if ele in punc:
            s = s.replace(ele, "")



    text_tokens = word_tokenize(s)

    tokens_without_sw = [word.lower() for word in text_tokens if not word.lower() in stopwords.words()]

    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in s:
        if ele in punc:
            s = s.replace(ele, "")

    s = ' '.join(tokens_without_sw)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in s:
        if ele in punc:
            s = s.replace(ele, "")
    df['title'] = df['title'].astype(str)
    df["count"]= df["title"].str.get(0)
    df = df.iloc[:20,:4]
    df['title'].map(lambda x: x.lower() if isinstance(x,str) else x)
    #df = df.iloc[:20,:4]
    #df['title'].map(lambda x: x.lower() if isinstance(x,str) else x)
    words = [x for x in s.split() if x != '']
    for j in range(len(df)):
        for i in words:
            df.at[j,i] = df['title'][j].count(i)
    return df
def plot_weighted_keywords(df):
    #df = df.append(df.sum().rename('Total'))
    #df.iloc[-1][4:-1].astype(int).nlargest(25).plot(kind="bar")
    #plt.rcParams["figure.figsize"] = (6,6)
    #plt.show()
    for i in df.columns[4:]:
        df[i] = df[i] * df['ups']
    df = df.append(df.sum().rename('Total'))
    df.iloc[-1][4:-1].astype(int).nlargest(25).plot(kind="bar")
    plt.rcParams["figure.figsize"] = (6,6)
    fig,ax=plt.subplots(figsize=(6,6))
    ax=sns.set(style="darkgrid")
    weighted_keywords = df.iloc[-1][4:-1]
    weighted_keywords = weighted_keywords.sort_values(ascending=False)
    weighted_keywords = weighted_keywords.drop(labels=[''])
    sns.barplot(weighted_keywords.index[:5],weighted_keywords.values[:5])
    canvas=FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return img
