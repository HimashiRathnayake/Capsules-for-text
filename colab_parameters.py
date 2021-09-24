# user parameters

Host = "colab" #@param ["colab", "AWS", "GCP"]
technique = "humor" #@param ["humor", "hate speech"]
EMBEDDING_SIZE = 300 #@param [50, 150, 200, 250, 300, 350, 400, 450, 500]
embedding_type = "fasttext" #@param ["fasttext","word2vec"]
experiment_no = "01" #@param [] {allow-input: true}
over_sampling_technique = "ROS" #@param ["", "ROS","ADASYN", "SMOTE", "BorderlineSMOTE"]
sampling_strategy = "0.5" #@param [] {allow-input: true}

# other parameters

if technique == "humor" :
  NO_OUTPUT_LAYERS = 2
  tag_set = ["Humorous", "Non-Humorous"]
elif technique == "hate speech":
  NO_OUTPUT_LAYERS = 3
  tag_set = ["Abusive", "Hate-Inducing", "Not offensive"]

# MAX_FEATURES = EMBEDDING_MATRIX.shape[0] #vocab_size
VERBOSITY = 1
VALIDATION_SPLIT = 0.1
# NB_EPOCHS = 10
FOLDS = 5 #10
# BATCH_SIZE = 32 # 64, 128
# NB_FILTERS = 200
# FILTER_LENGTH = 4 # test with 2,3,4,5
# HIDDEN_DIMS = NB_FILTERS * 2
# # MAX_LEN = max_length #275 #test with other values(only this value work for now)
# DROPOUT_VALUE_1 = 0.5 #0.8 #0.3
# DROPOUT_VALUE_2 = 0.5
# L2_REG= 0.01

# folder paths

folder_path = "/content/drive/Shareddrives/FYP/"
data_path = folder_path + "corpus/çompleted_draft.csv"

context = 5
word_embedding_path = folder_path + "Embedding models/" + embedding_type + '/' + str(EMBEDDING_SIZE) + "/embedding_" + embedding_type + "_" + str(EMBEDDING_SIZE)
word_embedding_keydvectors_path = folder_path + "Embedding models/" + embedding_type + '/' + str(EMBEDDING_SIZE) + "/keyed_vectors/" +  "embedding_" + embedding_type + "_" + str(EMBEDDING_SIZE)
embedding_matrix_path = folder_path + "Humor_HateSpeech detection/Implementation/embedding_matrix/"+embedding_type+'_'+str(EMBEDDING_SIZE)

# experiment_name = str(experiment_no) + "_"+ model_name +"_"+embedding_type+"_"+str(EMBEDDING_SIZE)+"_"+str(context)
# model_save_path = folder_path + "Humor_HateSpeech detection/Implementation/saved_models/"+technique+"/"+experiment_name+".hdf5"