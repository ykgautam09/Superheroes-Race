import pandas as pd
import re
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle

raw_data = pd.read_csv('./altered_data.csv', )
drop_index = raw_data[raw_data['race'] == 'Human'].index[150:]
raw_data.drop(index=drop_index, inplace=True)
# raw_data.dropna(inplace=True)
raw_data.sample(frac=1).reset_index(drop=True)
raw_data.bfill(inplace=True)
raw_data.ffill(inplace=True)
raw_data.drop_duplicates(inplace=True)

raw_data.sample(frac=1).reset_index(drop=True)

vectorizer = TfidfVectorizer(max_features=5000, analyzer='word', encoding='utf-8')
model = SVC()
lemmetizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))


def process_data(text_data):
    cleaned_data = []
    for sentence in text_data:
        sentence = re.sub('[^a-zA-Z]', ' ', sentence.lower())
        sentence = [i for i in sentence.split() if i not in stopwords]
        words = []
        for word in sentence:
            words.append(lemmetizer.lemmatize(word))
        cleaned_data.append(' '.join(words))
    return cleaned_data


clean_data = process_data(raw_data['history'])
vector = vectorizer.fit_transform(clean_data)
with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f, protocol=2)
model.fit(vector, raw_data['race'])
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f, protocol=2)
