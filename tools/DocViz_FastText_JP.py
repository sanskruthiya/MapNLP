from gensim import corpora, models
from glob import glob
from operator import itemgetter
from scipy.sparse.csgraph import connected_components
from statistics import mean
import fasttext
import hdbscan
import collections
import umap
import os
import sys
import csv
import string
import codecs
import re
import time
import MeCab
import pandas as pd
import numpy as np

#MeCab tokenizer
def mecab_tokenizer(tx, sw):
    token_list = []
    tagger = MeCab.Tagger('/usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    tagger.parse('')
    node = tagger.parseToNode(tx)
    while node:
        if not node.surface in sw:
            pos = node.feature.split(",")
            if not pos[0] in ["記号"]:
                token_list.append(node.surface)
        node = node.next
    return list(token_list)

#Feature-word extractor
def fword_extractor(num, docs):
    fword = docs[0][0]
    for c in range(num+1):
        if not c == 0:
            fword = fword + '|' + docs[c][0]
    return fword

#FastText vectorizer
def ft_vectorizer(tx, bsn):
    model_path = "02_models/cc.ja.300.bin"
    if not os.path.exists(model_path):
        model_txt_path = "02_models/" + bsn + "_ft_corpus" + ".txt"
        model_alt_path = "02_models/" + bsn + "_ft_corpus" + ".bin"
        with open(model_txt_path, 'w', encoding="utf_8") as f:
            text_corpus = ""
            for t in tx:
                text_corpus = " ".join(map(str, t))
                f.write(text_corpus+"\n")
        print("New FastText model was created at " + model_alt_path)
        model = fasttext.train_unsupervised(model_txt_path, model='cbow', dim=300)
        #model.save_model(model_alt_path)
    else:
        print("Pre-trained model " + model_path + " is used.")
        model = fasttext.load_model(model_path)
    
    ft_vec = np.zeros((df.shape[0], 300))
    #Prepairing an empty list for the model coverage rate
    coverage = []
    #Caluculating the average value of the vector of words by a document
    for i,doc in enumerate(tx):
        feature_vec = np.zeros(300) #Initializing the 300-dimension vector data to 0
        num_words = 0
        no_count = 0
        for word in doc:
            try:
                feature_vec += model[str(word)]
                num_words += 1
            except:
                no_count += 1
        feature_vec = feature_vec / num_words
        ft_vec[i] = feature_vec
        #Caluculating the word coverage rate of the model by each document
        cover_rate = num_words / (num_words + no_count)
        coverage.append(cover_rate)
    #Print overall word coverage rate of the model
    mean_coverage = round(mean(coverage)*100, 2)
    print("Word cover-rate: " + str(mean_coverage) + "%")
    return ft_vec

os.makedirs("01_dataset", exist_ok=True)
os.makedirs("02_models", exist_ok=True)
os.makedirs("03_processing", exist_ok=True)

#Open the input-data and store it into DataFrame
docs_file = input('Input a document-list file (.csv) in 01_dataset >> ')
docs_path = "01_dataset/" + str(docs_file)
if os.path.exists(docs_path):
    with codecs.open(docs_path, 'r', 'utf-8', 'ignore') as f:
        #df = pd.read_csv(f, delimiter='\t')
        df = pd.read_csv(f)
    base_name = os.path.splitext(os.path.basename(docs_file))[0] #The file name without the extension
else:
    print('Processing was cancelled due to an invalid input. Please enter a correct file name')
    sys.exit()

#Open stopword list
stops_file = input('Input a stopword-list file (.txt) in 02_models >> ')
stops_path = "02_models/" + str(stops_file)
if os.path.exists(stops_path):
    with codecs.open(stops_path, 'r', 'utf-8', 'ignore') as f:
        stopwords = f.read().splitlines()
else:
    print('The stopword file is not found. This process continues without stopwords.')
    stopwords = []

#Target column
df['text'] = df['description'].astype(str)
#Tokenizer
text_tokenizer = mecab_tokenizer

print("Input data size : " + str(len(df)))

#Remove https-links
df['text_clean'] = df.text.map(lambda x: re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', "", x))
#Remove numerals
df['text_clean'] = df.text_clean.map(lambda x: re.sub(r'\d+', '', x))
#Remove symbols
df['text_clean'] = df.text_clean.map(lambda x: re.sub(r'[「」。、,（）%#\$&\?\(\)~\.=\+\-\[\]\{\}\|\*]+', '', x))
#Remove specific noises (optional)
df['text_clean'] = df.text_clean.map(lambda x: re.sub(r'○.{0,20}　', ' ', x))
#Creating DataFrame for Token-list
df['text_tokens'] = df.text_clean.map(lambda x: text_tokenizer(x, stopwords))

#Creating dictionary and corpus
texts = df['text_tokens'].values
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=3, no_above=0.4)
corpus = [dictionary.doc2bow(text) for text in texts]

#Feature words extraction by tf-idf (option: min_df, max_df)
tfidf_model = models.TfidfModel(corpus, id2word=dictionary.token2id, normalize=False)
tfidf_corpus = list(tfidf_model[corpus])

method_name = "FT" #For the output file name
doc_vec = ft_vectorizer(df['text_tokens'], base_name)

tfidf_texts = []
for doc in tfidf_corpus:
    tfidf_text = []
    for word in doc:
        tfidf_text.append([dictionary[word[0]], word[1]])
        tfidf_text_sort = sorted(tfidf_text, reverse=True, key=itemgetter(1))
    tfidf_texts.append(tfidf_text_sort)

num_fword = 4 #Set the number of displayed feature words
fword_list = []
for i in range(len(tfidf_texts)):
    if len(tfidf_texts[i]) > num_fword-1:
        fword = fword_extractor(num_fword-1, tfidf_texts[i])
    else:
        fword = fword_extractor(len(tfidf_texts[i])-1, tfidf_texts[i])
    fword_list.append(fword)

#Storing in DataFrame
embedding_u = umap.UMAP(min_dist=0.1, n_neighbors=50, metric='euclidean', spread=1.0, init='spectral').fit_transform(doc_vec)
embedding = pd.DataFrame(embedding_u, columns=['x', 'y'])
embedding['uid'] = df['ID']
embedding['keywords'] = fword_list
embedding['size'] = df['size']
embedding['title'] = df['title']
embedding['date'] = df['date']
embedding['author'] = df['author']
embedding['affiliation'] = df['affiliation']
embedding['description'] = df['description']

x_coord = embedding_u[:, 0]
y_coord = embedding_u[:, 1]

#Clustering
clst_input = input('Input the number for min-size of a cluster (default=100) >> ')
try:
    clusters = int(clst_input)
except:
    clusters = 100 #set default value
    print("The number of cluster was set to 100, due to an invalid input")

type_input = input('Input type of clustering - eom:1 or leaf:2 >> ')
if type_input == "1":
    type_c = 'eom'
elif type_input == "2":
    type_c = 'leaf'
else:
    type_c = 'leaf'
    print("Type of the clustering is set to leaf due to an invalid input.")

clustering = hdbscan.HDBSCAN(cluster_selection_method=type_c, min_cluster_size=clusters, min_samples=10) #The lower the value, the less noise you’ll get
log_method = "HDBSCAN (minimum size of clusters: " + str(clusters) + " )"

log_list = []
label_list = []
log_list.append(log_method)

#Extract only XY coordinates from DataFrame for the clustering calculation
df_xy = embedding.loc[:, ['x', 'y']]

#Convert XY coordicates into numpy array
X = df_xy.to_numpy()
log_doc = "Documents : " + str(X.shape[0])
log_list.append(log_doc)
print(log_doc)

#Execution of clustering (using scikit-learn based modules)
clustering.fit(X)

#Store the label data
labels = clustering.labels_
num_labels = len(set(labels)) - (1 if -1 in labels else 0)
outliers = list(labels).count(-1)
log_label = "labels : " + str(num_labels) + ", outliers : " + str(outliers)
log_list.append(log_label)
print(log_label)

#Create DataFrame of ID and Label No
embedding_L1 = pd.DataFrame()
embedding_L1["id"] = df['ID']
embedding_L1["keywords"] = fword_list
embedding_L1["X"] = x_coord
embedding_L1["Y"] = y_coord
embedding_L1["label"] = labels

#Feature words extraction
num = 0
log_head = "area_id" + '\t' + "documents" + '\t' + "keywords" + '\t' + "X" + '\t' + "Y" + '\t' + "label"
log_list.append(log_head)
label_list.append(log_head)
while num < num_labels:
    f_list = []
    embedding_L2 = embedding_L1.query("label == @num")
    fwords = np.array(embedding_L2['keywords'])
    s = fwords.T

    for i in s:
        try:
            l = [x.strip() for x in i.split('|')]
            f_list.extend(l)
        except:
            pass
    
    c = collections.Counter(f_list)
    top_c = c.most_common(40)[0:39]
    doc_n = len(embedding_L2)
    x_mean = embedding_L2["X"].mean()
    y_mean = embedding_L2["Y"].mean()
    
    try:
        l_key1 = top_c[0][0] + "/" + top_c[1][0] + "/" + top_c[2][0]
        l_key2 = [k for k, v in top_c[3:] if v / doc_n > 0.4 and v / doc_n < 0.65]
        if len(l_key2) > 0:
            l_keys = l_key1 + "/" + l_key2[0]
        else:
            l_keys = l_key1
    except:
        l_keys = "-"
    
    log_fwords = str(num) + "\t" + str(doc_n) + "\t" + str(top_c) + "\t" + str(round(x_mean, 4)) + "\t" + str(round(y_mean, 4)) + "\t" + str(l_keys) #log the feature words of areas
    log_list.append(log_fwords)
    label_list.append(log_fwords)
    print(log_fwords)
    num += 1

#export the result
embedding.to_csv("03_processing/" + base_name + "_by" + method_name + ".csv", encoding="utf_8", index=False)

log_file = "log_" + str(num_labels) + ".txt"
with open("03_processing/" + base_name + "_by" + method_name + "_" + log_file, 'w', encoding="utf_8") as t:
    t.write("\n".join(log_list))

label_name = "label_" + str(num_labels) + ".tsv"
with open("03_processing/" + base_name + "_by" + method_name + "_" + label_name, 'w', encoding="utf_8") as t:
    t.write("\n".join(label_list))

print("Completed.")
