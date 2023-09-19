from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.20, random_state=1, shuffle = True)
train, val = train_test_split(train, test_size=0.20, random_state=1, shuffle = True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_vectorizer.fit(train['text'])

x_tfidf_train = tfidf_vectorizer.transform(train['text'])
x_tfidf_test = tfidf_vectorizer.transform(test['text'])

X_train = train['text']
X_test = test['text']
y_train = train['veracity']
y_test = test['veracity']
X_val = val['text']
y_val = val['veracity']