import network
import utils
from labelencoder import LabelEncoder
import codecs
import numpy as np
import pickle
from datetime import datetime
import argparse
import lasagne
import theano


embedding_vectors = np.load('embedding/vectors.npy')
with open('embedding/words.pl', 'rb') as handle:
    embedding_words = pickle.load(handle)
embedd_dim = np.shape(embedding_vectors)[1]
unknown_embedd = np.load('embedding/unknown.npy')
word_end = "##WE##"

batch_size = 10
output_dir = 'output/ner'

def load_conll_data(path):
    word_sentences = []
    pos_sentences = []
    chunk_sentences = []
    label_sentences = []
    words = []
    poss = []
    chunks = []
    labels = []
    with codecs.open(path, 'r', 'utf-8') as file:
        for line in file:
            if line.strip() == "":
                word_sentences.append(words[:])
                pos_sentences.append(poss[:])
                chunk_sentences.append(chunks[:])
                label_sentences.append(labels[:])
                words = []
                poss = []
                chunks = []
                labels = []
            else:
                tokens = line.strip().split()
                word = tokens[0]
                pos = tokens[1]
                chunk = tokens[2]
                label = tokens[3]
                words.append(word)
                poss.append(pos)
                chunks.append(chunk)
                labels.append(label)
    return word_sentences, pos_sentences, chunk_sentences, label_sentences

def create_data_2_test(test_path, max_length, max_char_length, alphabet_pos, alphabet_chunk, alphabet_label, alphabet_char):
    # word_sentences_train, pos_sentences_train, chunk_sentences_train, label_sentences_train = load_conll_data(train_path)
    # word_sentences_dev, pos_sentences_dev, chunk_sentences_dev, label_sentences_dev = load_conll_data(dev_path)
    word_sentences_test, pos_sentences_test, chunk_sentences_test, label_sentences_test = load_conll_data(test_path)
    # max_length_train = utils.get_max_length(word_sentences_train)
    # max_length_dev = utils.get_max_length(word_sentences_dev)
    # max_length_test = utils.get_max_length(word_sentences_test)
    # max_length = max(max_length_train, max_length_dev, max_length_test)

    # pos_sentences_id_train, alphabet_pos = utils.map_string_2_id_open(pos_sentences_train, 'pos')
    # alphabet_pos.save('pre-trained-model/ner', name='alphabet_pos')
    # pos_sentences_id_dev = utils.map_string_2_id_close(pos_sentences_dev, alphabet_pos)
    pos_sentences_id_test = utils.map_string_2_id_close(pos_sentences_test, alphabet_pos)
    # chunk_sentences_id_train, alphabet_chunk = utils.map_string_2_id_open(chunk_sentences_train, 'chunk')
    # alphabet_chunk.save('pre-trained-model/ner', name='alphabet_chunk')
    # chunk_sentences_id_dev = utils.map_string_2_id_close(chunk_sentences_dev, alphabet_chunk)
    chunk_sentences_id_test = utils.map_string_2_id_close(chunk_sentences_test, alphabet_chunk)
    # label_sentences_id_train, alphabet_label = utils.map_string_2_id_open(label_sentences_train, 'ner')
    # alphabet_label.save('pre-trained-model/ner', name='alphabet_label')
    # label_sentences_id_dev = utils.map_string_2_id_close(label_sentences_dev, alphabet_label)
    label_sentences_id_test = utils.map_string_2_id_close(label_sentences_test, alphabet_label)
    # word_train, label_train, mask_train = \
    #     utils.construct_tensor_word(word_sentences_train, label_sentences_id_train, unknown_embedd, embedding_words,
    #                                 embedding_vectors, embedd_dim, max_length)
    # word_dev, label_dev, mask_dev = \
    #     utils.construct_tensor_word(word_sentences_dev, label_sentences_id_dev, unknown_embedd, embedding_words,
    #                                 embedding_vectors, embedd_dim, max_length)
    word_test, label_test, mask_test = \
        utils.construct_tensor_word(word_sentences_test, label_sentences_id_test, unknown_embedd, embedding_words,
                                    embedding_vectors, embedd_dim, max_length)
    # pos_train = utils.construct_tensor_onehot(pos_sentences_id_train, max_length, alphabet_pos.size())
    # pos_dev = utils.construct_tensor_onehot(pos_sentences_id_dev, max_length, alphabet_pos.size())
    pos_test = utils.construct_tensor_onehot(pos_sentences_id_test, max_length, alphabet_pos.size())
    # chunk_train = utils.construct_tensor_onehot(chunk_sentences_id_train, max_length, alphabet_chunk.size())
    # chunk_dev = utils.construct_tensor_onehot(chunk_sentences_id_dev, max_length, alphabet_chunk.size())
    chunk_test = utils.construct_tensor_onehot(chunk_sentences_id_test, max_length, alphabet_chunk.size())
    # word_train = np.concatenate((word_train, pos_train), axis=2)
    # word_train = np.concatenate((word_train, chunk_train), axis=2)
    # word_dev = np.concatenate((word_dev, pos_dev), axis=2)
    # word_dev = np.concatenate((word_dev, chunk_dev), axis=2)
    word_test = np.concatenate((word_test, pos_test), axis=2)
    word_test = np.concatenate((word_test, chunk_test), axis=2)
    # alphabet_char = LabelEncoder('char')
    # alphabet_char.get_index(word_end)
    # index_sentences_train, max_char_length_train = utils.get_character_indexes(word_sentences_train, alphabet_char)
    alphabet_char.close()
    # char_embedd_table = utils.build_char_embedd_table(char_embedd_dim, alphabet_char)
    # alphabet_char.save('pre-trained-model/ner', name='alphabet_char')
    # index_sentences_dev, max_char_length_dev = utils.get_character_indexes(word_sentences_dev, alphabet_char)
    index_sentences_test, max_char_length_test = utils.get_character_indexes(word_sentences_test, alphabet_char)
    # max_char_length = max(max_char_length_train, max_char_length_dev, max_char_length_test)
    # char_train = utils.construct_tensor_char(index_sentences_train, max_length, max_char_length, alphabet_char)
    # char_dev = utils.construct_tensor_char(index_sentences_dev, max_length, max_char_length, alphabet_char)
    char_test = utils.construct_tensor_char(index_sentences_test, max_length, max_char_length, alphabet_char)
    # num_labels = alphabet_label.size() - 1
    # num_data, _, embedd_dim_concat = word_train.shape
    # print(np.shape(word_train))
    # print(np.shape(word_dev))
    # print(np.shape(word_test))
    # print(np.shape(char_train))
    # print(np.shape(char_dev))
    # print(np.shape(char_test))
    # print(np.shape(mask_train))
    # print(np.shape(mask_dev))
    # print(np.shape(mask_test))
    # print(np.shape(label_train))
    # print(np.shape(label_dev))
    # print(np.shape(label_test))
    # print(word_train[-1])
    return word_test, char_test, mask_test, label_test

def load_config(config_file):
    config = dict()
    with codecs.open(config_file, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            config[line[0]] = line[1]
    max_sent_length = int(config['max_sent_length'])
    max_char_length = int(config['max_char_length'])
    num_labels = int(config['num_labels'])
    dropout = config['dropout']
    if dropout == 'True':
        dropout = True
    elif dropout == 'False':
        dropout = False
    num_filters = int(config['num_filters'])
    num_units = int(config['num_units'])
    grad_clipping = float(config['grad_clipping'])
    peepholes = config['peepholes']
    if peepholes == 'True':
        peepholes = True
    elif peepholes == 'False':
        peepholes = False
    embedd_dim_concat = int(config['embedd_dim_concat'])

    alphabet_pos = LabelEncoder('alphabet_pos')
    alphabet_pos.load('pre-trained-model/ner')
    alphabet_chunk = LabelEncoder('alphabet_chunk')
    alphabet_chunk.load('pre-trained-model/ner')
    alphabet_label = LabelEncoder('alphabet_label')
    alphabet_label.load('pre-trained-model/ner')
    alphabet_char = LabelEncoder('alphabet_char')
    alphabet_char.load('pre-trained-model/ner')

    char_embedd_dim = int(config['char_embedd_dim'])
    char_embedd_table = utils.build_char_embedd_table(char_embedd_dim, alphabet_char)
    return max_sent_length, max_char_length, num_labels, dropout, num_filters, num_units, grad_clipping, peepholes, \
           char_embedd_dim, alphabet_pos, alphabet_chunk, alphabet_label, alphabet_char, char_embedd_table, embedd_dim_concat

def set_weights(filename, model):
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(model, param_values)

if __name__ == '__main__':
    test_dir = 'data/ner/ner_test.txt'
    start_time = datetime.now()
    print('Loading config...')
    max_sent_length, max_char_length, num_labels, dropout, num_filters, num_units, grad_clipping, peepholes, \
    char_embedd_dim, alphabet_pos, alphabet_chunk, alphabet_label, alphabet_char, char_embedd_table, embedd_dim_concat = \
        load_config('pre-trained-model/ner/config.ini')

    print('Loading data...')
    word_test, char_test, mask_test, label_test = \
        create_data_2_test(test_dir, max_sent_length, max_char_length, alphabet_pos, 
                           alphabet_chunk, alphabet_label, alphabet_char)

    print('Building model...')
    ner_model, input_var, target_var, mask_var, char_input_var = \
        network.build_model(embedd_dim_concat, max_sent_length, max_char_length, alphabet_char.size(), char_embedd_dim,
                            num_labels, dropout, num_filters, num_units, grad_clipping, peepholes, char_embedd_table)

    set_weights('pre-trained-model/ner/weights.npz', ner_model)
    energies = lasagne.layers.get_output(ner_model, deterministic=True)
    prediction = utils.crf_prediction(energies)
    prediction_fn = theano.function([input_var, mask_var, char_input_var], [prediction])
    
    for batch in utils.iterate_minibatches(word_test, label_test, masks=mask_test, char_inputs=char_test,
                                                   batch_size=batch_size):
        inputs, targets, masks, char_inputs = batch
        predictions = prediction_fn(inputs, masks, char_inputs)
        print(predictions)
        utils.output_predictions(predictions[0], targets, masks, output_dir + '/test_ner', alphabet_label,
                                         is_flattened=False)
