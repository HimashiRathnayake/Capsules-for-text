from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,precision_recall_fscore_support
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils

import ensemble_capsule_network
from config import Config
from preprocessing import text_preprocessing, load_word_embedding_matrix, generate_embedding_matrix
from network import get_capsule_network_model
from methods_colab import load_data, apply_oversampling
from parameters_colab import *

all_data = load_data()

comments_text, labels = text_preprocessing(all_data)
t = Tokenizer()
t.fit_on_texts(comments_text)
vocab_size = len(t.word_index) + 1
print(vocab_size)

encoded_docs = t.texts_to_sequences(comments_text)
max_length = 30
# max_length = len(max(encoded_docs, key=len))
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
padded_docs = np.array(padded_docs)
comment_labels = np.array(labels)
comment_labels = pd.get_dummies(comment_labels).values

print("Shape of all comments: ", padded_docs.shape)
print("Shape of labels: ", comment_labels.shape)

X_train, X_test, Y_train, Y_test = train_test_split(padded_docs, comment_labels, test_size=0.1, random_state=42,
                                                    shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(padded_docs, comment_labels, test_size=0.1, random_state=42,
                                                    shuffle=True)
x_train, y_train = apply_oversampling(x_train, y_train);
x_test = X_test
y_test = Y_test

print("Train lables shape: ", y_train.shape)

# generate embedding matrix
# embedding_matrix = generate_embedding_matrix(word_embedding_keydvectors_path, embedding_matrix_path, vocab_size,
#                                              EMBEDDING_SIZE, t)

# load embedding matrix
embedding_matrix = load_word_embedding_matrix(embedding_matrix_path)

# print(embedding_matrix[1])
config = Config(
    seq_len=max_length,
    num_classes=NO_OUTPUT_LAYERS,
    vocab_size=vocab_size,
    embedding_size=EMBEDDING_SIZE,
    dropout_rate=0.8,
    x_train=x_train,
    y_train=y_train,
    x_test=x_val,
    y_test=y_val,
    pretrain_vec=embedding_matrix)

model = ensemble_capsule_network.ensemble_capsule_network(config)
# model = get_capsule_network_model(config)

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=50)

predictions = model.predict(x_test)
labels = np.argmax(y_test, axis=1)
predictions = np.argmax(predictions, axis=1)

# classification_report
report_print = classification_report(labels, predictions, digits=4)
print(report_print)

print("Accuracy: ", accuracy_score(labels, predictions))
print("Precision: ", precision_score(labels, predictions, average='weighted'))
print("Recall: ", recall_score(labels, predictions, average='weighted'))
print("F1-Score: ", f1_score(labels, predictions, average='weighted'))
print("Macro Precision: ", precision_score(labels, predictions, average='macro'))
print("Macro Recall: ", recall_score(labels, predictions, average='macro'))
print("Macro F1-Score: ", f1_score(labels, predictions, average='macro'))