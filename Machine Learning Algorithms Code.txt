### Naive Bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Application of TF-IDF on the training texts
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Creation and training of the Multinomial Naive Bayes model
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)
# Application of TF-IDF to the test texts
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Classification of the test texts with the model
y_pred = naive_bayes.predict(X_test_tfidf)
# Evaluation of performance"
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (Accuracy):", accuracy)
# Printing the classification report
print(classification_report(y_test, y_pred))


### KNN (K nearest neighbors)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
data = pd.read_csv(r'C:\Users\Desktop\corpus.csv', encoding='utf-16')
data = data.dropna()  
X = data['Text								']  
y = data['type'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=10000) 
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
knn_classifier = KNeighborsClassifier(n_neighbors=5)  
knn_classifier.fit(X_train_tfidf, y_train)
y_pred = knn_classifier.predict(X_test_tfidf)
print("Accuracy (Accuracy):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



### CART Algorithm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv(r'C:\Users\Desktop\yourcorpus.csv', encoding='utf-16')
data = data.dropna()  # We delete the lines with empty values.
X = data['Text								']
y = data['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

cart_classifier = DecisionTreeClassifier(random_state=42)
cart_classifier.fit(X_train_tfidf, y_train)
y_pred = cart_classifier.predict(X_test_tfidf)
print("Accuracy (Accuracy):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


### Decision Trees


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
data = pd.read_csv(r'C:\Users\Desktop\yourcorpus.csv', encoding='utf-16')
data = data.dropna()  
X = data['Text								'] 
y = data['type']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_tfidf, y_train)
y_pred = dt_classifier.predict(X_test_tfidf)
print("Accuracy (Accuracy):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



### SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
# Reads the data from a CSV file
df = pd.read_csv(r'C:\Users\xristos\Desktop\corpus.csv', encoding='utf-16')
# We define the variables that contain the features (X) and the labels (y)
x = df['Text Here								']  
# Replace 'Κείμενο' with the name of the column that contains the texts.
df['Text Here							'].fillna('', inplace=True)
y = df['type']  
  # Replace 'type' with the name of the column that contains the labels.
# We split the data into a training set (85%) and a testing set (15%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
# Creating the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
# Applying TF-IDF to the training texts
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
# Creating the SVM classifier
svm_classifier = SVC(kernel='linear')
# Training the classifier on the training set
svm_classifier.fit(x_train_tfidf, y_train)
# Applying TF-IDF to the test texts 
x_test_tfidf = tfidf_vectorizer.transform(x_test)
# Categorization of the test texts
y_pred = svm_classifier.predict(x_test_tfidf)
# Calculation of performance metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy (Accuracy):", accuracy)
print("Classification Report:")
print(report)






### Word2Vec (Logistic Regression)
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
# We read the data from the CSV file.
df = pd.read_csv(r'C:\Users\xristos\Desktop\corpus.csv', encoding='utf-16')
# We define the variables that contain the features (X)
x = df['Text Here							'].apply(lambda text: text.strip() if isinstance(text, str) else '')
df['Text Here							'].fillna('', inplace=True)
y = df['type']
# We split the texts into words.
x = x.apply(lambda text: word_tokenize(text.lower()))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Training the Word2Vec model.
model = Word2Vec(x, vector_size=100, window=5, min_count=1, sg=0)
# We save the model for future use.
model.save("word2vec_model")
# To load the model for later use.
model = Word2Vec.load("word2vec_model")
# Extraction of word vectors while maintaining the same length.
def get_word_vector(word):
    if word in model.wv:
        return model.wv[word]
    else:
        return np.zeros(100)  # We use zero vectors for words that are not found in the model.
def text_to_average_vector(text):
    word_vectors = [get_word_vector(word) for word in text if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(100)
# Conversion of training texts to average vectors.
average_vectors_train = [text_to_average_vector(text) for text in X_train]
# Training a Logistic Regression model.
logistic_model = LogisticRegression()
logistic_model.fit(average_vectors_train, y_train)
# Convert the test texts into average vectors and make predictions.
average_vectors_test = [text_to_average_vector(text) for text in X_test]
predictions = logistic_model.predict(average_vectors_test)
# Calculation of performance metrics.
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print("Accuracy (Accuracy):", accuracy)
print("Classification Report.:")
print(report)




### SVC

# Importing the necessary libraries.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Reading the data from the CSV file.
df = pd.read_csv(r'C:\Users\xristos\Desktop\corpus.csv', encoding='utf-16')
# We define the variable x that contains the texts from the DataFrame.
x = df['Text Here							'].apply(lambda text: text.strip() if isinstance(text, str) else '')
df['Text Here								'].fillna('', inplace=True)
y = df['type']
# Splitting the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
y = df['Type']
# Creating the TF-IDF vectorizer.
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
# Applying TF-IDF to the training features.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Creating the SVM classifier.
svm_classifier = SVC(kernel='linear')
# Training the classifier on the training data.
svm_classifier.fit(X_train_tfidf, y_train)
# Applying TF-IDF to the test features.
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Classifying the test data.
y_pred = svm_classifier.predict(X_test_tfidf)
# Calculating performance metrics.
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy (Accuracy):", accuracy)
print("Classification report:")
print(report)



### Neural Network

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from tensorflow.sparse import SparseTensor
from sklearn.utils import shuffle
import tensorflow as tf
tfidf_vectorizer = TfidfVectorizer()
df = pd.read_csv(r'C:\Users\Desktop\your corpus.csv', encoding='utf-16')
unique_labels = df['type'].unique()
print(unique_labels)
label_counts = df['type'].value_counts()
print(label_counts)
num_unique_labels = len(unique_labels)
print("Total number of unique labels:", num_unique_labels)
X = df['Text Here								']
df['Text Here								'].fillna('', inplace=True)
y = df['type']
X_tfidf = tfidf_vectorizer.fit_transform(X)
X_train_tfidf_sparse = tf.sparse.reorder(tf.sparse.SparseTensor(X_train_tfidf.indices, X_train_tfidf.data, X_train_tfidf.shape))
X_test_tfidf_sparse = tf.sparse.reorder(tf.sparse.SparseTensor(X_test_tfidf.indices, X_test_tfidf.data, X_test_tfidf.shape))
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.1, random_state=42)
num_features = X_train.shape[1]
num_classes = len(unique_labels)
model = Sequential()
model.add(Dense(64, input_dim=num_features, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# We define the optimization and loss function
optimizer = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Training the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
model.fit(X_train_tfidf_sparse, y_train, epochs=10, batch_size=64, validation_data=(X_test_tfidf_sparse, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
print(sparse.issparse(X_train))  # Should return True
print(sparse.issparse(X_test))   # Should return True


### Random Forest

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r'C:\Users\Desktop\yourcorpus.csv', encoding='utf-16')
X = df['Text 								']
df['Text								'].fillna('', inplace=True)
y = df['type']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Ακρίβεια: {accuracy}')
print(classification_report(y_test, y_pred))

