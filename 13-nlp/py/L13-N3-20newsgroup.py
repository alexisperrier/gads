# L13 N3

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier


from sklearn.pipeline import Pipeline
categories = ['sci.space','comp.graphics', 'sci.med', 'rec.motorcycles', 'rec.sport.baseball']
twenty_train = fetch_20newsgroups(subset='train',  categories=categories, shuffle=True, random_state=42)

twenty_train.target_names
len(twenty_train.data)

# test data
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)


#
# Count words
count_vect = CountVectorizer(stop_words = 'english', max_features= 1000)
X_train_counts = count_vect.fit_transform(twenty_train.data)

# Frequency tf-idf

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)


for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

np.mean(predicted == twenty_test.target)

# PIpeline

text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english', max_features= 1000),
                   ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english', max_features= 1000),
                   ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
])


from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted , target_names=twenty_test.target_names))



