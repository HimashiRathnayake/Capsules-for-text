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



print("Train lables shape: ", y_train.shape)

# generate embedding matrix
# embedding_matrix = generate_embedding_matrix(word_embedding_keydvectors_path, embedding_matrix_path, vocab_size,
#                                              EMBEDDING_SIZE, t)

# load embedding matrix
embedding_matrix = load_word_embedding_matrix(embedding_matrix_path)

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

  X_train, X_test, Y_train, Y_test = train_test_split(padded_docs, comment_labels, test_size=0.1, random_state=42,
                                                    shuffle=True)
  x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42,
                                                      shuffle=True)
  x_train, y_train = apply_oversampling(x_train, y_train);
  x_test = X_test
  y_test = Y_test

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  model = build_model()
  model, his = Train_Model(model, inputs[train], targets[train], cross_validation=True)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)

  y_val_pred = model.predict(inputs[test])
  y_val_pred_cat = (np.asarray(y_val_pred)).round()
  y_val = targets[test]

  # Get performance metrics(macro averages) after each fold
  macro_f1, macro_precision, macro_recall = f1_score(y_val, y_val_pred_cat, average='macro'), precision_score(y_val, y_val_pred_cat, average='macro'), recall_score(y_val, y_val_pred_cat, average='macro')

  print(f"""Score for fold {fold_no}:
    {model.metrics_names[0]} of {scores[0]}; 
    {model.metrics_names[1]} of {scores[1]*100}% ;
    {model.metrics_names[2]} of {scores[2]*100}% ;
    {model.metrics_names[3]} of {scores[3]*100}% ;
    {model.metrics_names[4]} of {scores[4]*100}% ;
    macro precision of {macro_precision*100}% ;
    macro recall of {macro_recall*100}% ;
    macro f1 of {macro_f1*100}% ;
    """)
  
  loss_per_fold.append(scores[0])
  acc_per_fold.append(scores[1])
  precision_per_fold.append(scores[2])
  recall_per_fold.append(scores[3])
  f1_per_fold.append(scores[4])

  macro_f1_per_fold.append(round(macro_f1, 6))
  macro_precision_per_fold.append(round(macro_precision, 6))
  macro_recall_per_fold.append(round(macro_recall, 6))

  # Increase fold number
  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f"""> Fold {i+1} - 
  Loss: {loss_per_fold[i]} - 
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
print(f'> Loss: {np.mean(loss_per_fold)}')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Precision: {np.mean(precision_per_fold)}')
print(f'> Recall: {np.mean(recall_per_fold)}')
print(f'> F1: {np.mean(f1_per_fold)}')
print(f'> Macro Precision: {np.mean(macro_precision_per_fold)}')
print(f'> Macro Recall: {np.mean(macro_recall_per_fold)}')
print(f'> Macro F1: {np.mean(macro_f1_per_fold)}')
print('------------------------------------------------------------------------')