import pickle
from preprocess import process_data

with open('model.pickle', 'rb')  as f:
    model = pickle.load(f)
with open('vectorizer.pickle', 'rb')  as f:
    vectorizer = pickle.load(f)


def get_hero_race(history):
    clean_text = process_data([history])
    vector = vectorizer.transform(clean_text)
    result = model.predict(vector)
    print(result)
    return result[0].strip()


if __name__ == '__main__':
    get_hero_race()
