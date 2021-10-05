%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

data = fetch_20newsgroups()
data.target_names

categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

# data for train
train = fetch_20newsgroups(subset='train', categories=categories)


# test
test = fetch_20newsgroups(subset='test', categories=categories)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# fit model
model.fit(train.data, train.target)

pred = model.predict(test.data)

print(test.data, test.target)
print(train.data, train.target)

from sklearn.metrics import classification_report, accuracy_score
acc = classification_report(test.target, pred)
print(acc)


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

predict_category("""Seasonal summer rains have done little to offset drought conditions gripping the western United States, with California and Nevada seeing record July heat and moderate-to-exceptional drought according to the National Oceanic and Atmospheric Administration (NOAA). Now, new NASA research is showing how drought in the region is expected to change in the future, providing stakeholders with crucial information for decision making.

The study, published in the peer-reviewed journal, Earth’s Future, was led by scientists at NASA’s Goddard Institute for Space Studies (GISS) and funded by NOAA’s Climate Program Office and NASA’s Modeling, Analysis and Prediction (MAP) Program. It found that the western United States is headed for prolonged drought conditions whether greenhouse gas emissions continue to climb or are aggressively reined in. """)


for i in range(len(pred)):
    print(categories[pred[i]])
