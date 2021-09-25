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
from colab_methods import load_data, apply_oversampling
from colab_parameters import *
import hyperopt
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
import warnings
from csv import writer

all_data = load_data()

comments_text, labels = text_preprocessing(all_data)
t = Tokenizer()
t.fit_on_texts(comments_text)
vocab_size = len(t.word_index) + 1
print(vocab_size)

encoded_docs = t.texts_to_sequences(comments_text)
# max_length = 30
max_length = len(max(encoded_docs, key=len))
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
padded_docs = np.array(padded_docs)
comment_labels = np.array(labels)
comment_labels = pd.get_dummies(comment_labels).values

embedding_matrix = load_word_embedding_matrix(embedding_matrix_path)

np.random.seed(0)

space = {
    # 'init_lr': hp.choice('init_lr', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.95]), # 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
    'dropout_ratio': hp.choice('dropout_ratio', [0.8]), # 0.3, 0.5
    'batch_size': hp.choice('batch_size', [32]),
    'epochs': hp.choice('epochs', [50]),
    'l2': hp.choice('l2', [0.002]), # 0.001, 0.01, 0.02
    'optimizer': hp.choice('optimizer', ['SGD']) # 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
}

x_train, x_test, y_train, y_test = train_test_split(padded_docs, comment_labels, test_size=VALIDATION_SPLIT*2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VALIDATION_SPLIT, random_state=0)
x_train, y_train = apply_oversampling(x_train, y_train);

trials = Trials()

def objective(params):

    print({'params': params})

    config = Config(
        seq_len=max_length,
        num_classes=NO_OUTPUT_LAYERS,
        vocab_size=vocab_size,
        embedding_size=EMBEDDING_SIZE,
        # init_lr=params["init_lr"],
        dropout_ratio=params["dropout_ratio"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        l2=params["l2"],
        optimizer=params["optimizer"],
        x_train=x_train,
        y_train=y_train,
        x_test=x_val,
        y_test=y_val,
        pretrain_vec=embedding_matrix
      )

    model = ensemble_capsule_network.ensemble_capsule_network(config)
    model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=config.epochs, batch_size=config.batch_size)

    predictions = model.predict(x_test)
    labels = np.argmax(y_test, axis=1)
    predictions = np.argmax(predictions, axis=1)

    ret = {'loss': - f1_score(labels, predictions, average='macro'),'params': params, 'status': STATUS_OK}
    print("Current Trial", ret)
    print("Completed Trials:", trials.results)

    with open('/content/drive/Shareddrives/FYP/optimize/result.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(["loss", "batch_size", "dropout_ratio", "epochs", "l2", "optimizer"])
        writer_object.writerow([
          ret["loss"], 
          ret["params"]["batch_size"],
          ret["params"]["dropout_ratio"],
          ret["params"]["epochs"],
          ret["params"]["l2"],
          ret["params"]["optimizer"]
          ])
        f_object.close()


    return ret

# Run optimization - Random search.
def tune_hyperparameres():
  best = fmin(
      fn = objective, 
      space = space, 
      algo = tpe.rand.suggest,
      max_evals = 1, 
      trials = trials, 
      verbose = 0
  )
  print(best)
  print(hyperopt.space_eval(space, best))

tune_hyperparameres()

