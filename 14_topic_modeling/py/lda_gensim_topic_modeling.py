import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from gensim import corpora, models, similarities


from nltk import word_tokenize
from nltk.corpus import stopwords
import string


n_clusters = 5
categories   = ['sci.space','comp.graphics', 'sci.med', 'rec.motorcycles', 'rec.sport.baseball']
dataset = fetch_20newsgroups(subset='train',  categories=categories, shuffle=True, random_state=42)

stop = set(stopwords.words('english'))
exclude = list(string.punctuation)
def cleanup(raw):
    # lowercase
    raw = raw.lower()
    # ponctuation
    raw = ''.join(ch for ch in raw if ch not in exclude)
    # tokenize
    raw = word_tokenize(raw)
    # stop words
    raw = [w for w in raw if w not in stop]
    # at least 3 letters
    raw = [w for w in raw if len(w) > 2]

    return raw

tokenized = [ cleanup(raw) for raw in dataset.data[0:100] ]

# Dictionnary
dictionary = corpora.Dictionary(tokenized)
dictionary.save('reuters.dict')  # store the dictionary, for future reference
print(dictionary)
print(dictionary.token2id)

# Corpus
corpus = [dictionary.doc2bow(text) for text in tokenized]
corpora.MmCorpus.serialize('reuters.mm', corpus)  # store to disk, for later use
for c in corpus:
    print(c)

# TfIdf
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]

for doc in corpus_tfidf:
    print(doc)

# LSI
# initialize an LSI transformation
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)
corpus_lsi = lsi[corpus_tfidf]

lsi.print_topics(5, num_words = 15)

# see which doc belongs to which topic
for doc in corpus_lsi:
    print(doc)

# save
lsi.save('reuters.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('reuters.lsi')

# remove words that appear only once
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1

# texts = [[token for token in text if frequency[token] > 1] for text in texts]


