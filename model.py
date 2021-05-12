import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('news-data.csv')

df = df.drop_duplicates()

df = df.reset_index()


special_char_remover = re.compile('[/(){}\[\]\@,:;?$''""]')
extra_symbol_removel = re.compile('[^0-9a-z #+_]')
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = special_char_remover.sub(' ',text)
    text = extra_symbol_removel.sub('', text)
    text = ' '.join((word) for word in text.split() if word not in stop_words)
    return text
df['text'] = df['text'].apply(clean_text)

tfidf = TfidfVectorizer(max_features= 1500, ngram_range=(1, 2))

X = tfidf.fit_transform(df['text']).toarray()

y = df.category

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 3)

lr = LogisticRegression()

lr.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)

accuracy_score(y_pred1,y_test)


f = open('lr.pickle', 'wb')
pickle.dump(lr, f)
f.close()

f = open('tfidf.pickle', 'wb')
pickle.dump(tfidf, f)
f.close()






