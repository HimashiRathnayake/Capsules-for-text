from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,precision_recall_fscore_support
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils

import ensemble_capsule_network
from config import Config
from preprocessing import text_preprocessing, load_word_embedding_matrix, generate_embedding_matrix
from network import get_capsule_network_model
from colab_methods import load_data, apply_oversampling
from colab_parameters import *

all_data = load_data()

comments_text, labels = text_preprocessing(all_data)
t = Tokenizer()
t.fit_on_texts(comments_text)
vocab_size = len(t.word_index) + 1
print(vocab_size)

encoded_docs = t.texts_to_sequences(comments_text)
max_length = len(max(encoded_docs, key=len))
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
padded_docs = np.array(padded_docs)
comment_labels = np.array(labels)
comment_labels = pd.get_dummies(comment_labels).values

print("Shape of all comments: ", padded_docs.shape)
print("Shape of labels: ", comment_labels.shape)

# generate embedding matrix
# embedding_matrix = generate_embedding_matrix(word_embedding_keydvectors_path, embedding_matrix_path, vocab_size,
#                                              EMBEDDING_SIZE, t)

# load embedding matrix
embedding_matrix = load_word_embedding_matrix(embedding_matrix_path)

# Define per-fold score containers
acc_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []
macro_f1_per_fold = []
macro_precision_per_fold = []
macro_recall_per_fold = []

kfold = KFold(n_splits=FOLDS, shuffle=True)

fold_no = 1
inputs = padded_docs
targets = comment_labels

for train, test in kfold.split(inputs, targets):

  x_train, x_val, y_train, y_val = train_test_split(inputs[train], targets[train], test_size=0.1, random_state=0, shuffle=True)
  x_test = inputs[test]
  y_test = targets[test]

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  
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
  model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=100)

  # validate model

  predictions = model.predict(x_test)
  labels = np.argmax(y_test, axis=1)
  predictions = np.argmax(predictions, axis=1)

  # Get performance metrics(macro averages) after each fold
  accuracy = accuracy_score(labels, predictions)
  precision = precision_score(labels, predictions, average='weighted')
  recall = recall_score(labels, predictions, average='weighted')
  f1 = f1_score(labels, predictions, average='weighted')
  macro_precision = precision_score(labels, predictions, average='macro')
  macro_recall = recall_score(labels, predictions, average='macro')
  macro_f1 = f1_score(labels, predictions, average='macro')

  print(f"""Score for fold {fold_no}:
    accuracy of {accuracy*100}% ;
    precision of {precision*100}% ;
    recall of {recall*100}% ;
    f1 of {f1*100}% ;
    macro precision of {macro_precision*100}% ;
    macro recall of {macro_recall*100}% ;
    macro f1 of {macro_f1*100}% ;
    """)
  
  acc_per_fold.append(round(accuracy, 6))
  precision_per_fold.append(round(precision, 6))
  recall_per_fold.append(round(recall, 6))
  f1_per_fold.append(round(f1, 6))
  macro_precision_per_fold.append(round(macro_precision, 6))
  macro_recall_per_fold.append(round(macro_recall, 6))
  macro_f1_per_fold.append(round(macro_f1, 6))

  # Increase fold number
  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f"""> Fold {i+1} - 
  Accuracy: {acc_per_fold[i]}% - 
  Precesion: {precision_per_fold[i]}% - 
  Recall: {recall_per_fold[i]}% - 
  F1: {f1_per_fold[i]}%
  Macro Precision: {macro_precision_per_fold[i]}%
  Macro Recall: {macro_recall_per_fold[i]}%
  Macro F1: {macro_f1_per_fold[i]}%
  """)
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Precision: {np.mean(precision_per_fold)}')
print(f'> Recall: {np.mean(recall_per_fold)}')
print(f'> F1: {np.mean(f1_per_fold)}')
print(f'> Macro Precision: {np.mean(macro_precision_per_fold)}')
print(f'> Macro Recall: {np.mean(macro_recall_per_fold)}')
print(f'> Macro F1: {np.mean(macro_f1_per_fold)}')
print('------------------------------------------------------------------------')