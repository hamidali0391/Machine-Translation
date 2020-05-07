import collections

import helper
import numpy as np
import project_tests as tests

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import LSTM


def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Build the layers
    input_layer=InputLayer(input_shape[1:])
    rnn=GRU(64,return_sequences=True)
    logits=TimeDistributed(Dense(french_vocab_size,activation='softmax'))
    
    # TODO: Implement
    learning_rate=1e-3
    model=Sequential()
    model.add(input_layer)
    model.add(rnn)
    model.add(logits)
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
   
    return model

def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    #model_input=Input(input_shape[1:])
    embed_layer=Embedding(french_vocab_size,64,input_length=input_shape[1])
    rnn=GRU(64,return_sequences=True)
    logits=TimeDistributed(Dense(french_vocab_size,activation='softmax'))
    
    # TODO: Implement
    learning_rate=1e-3
    model=Sequential()
    model.add(embed_layer)
    model.add(rnn)
    model.add(logits)
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model

def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
   # TODO: Build the layers
    input_layer=InputLayer(input_shape[1:])
    rnn=Bidirectional(GRU(64,return_sequences=True))
    logits=TimeDistributed(Dense(french_vocab_size,activation='softmax'))
    
    # TODO: Implement
    learning_rate=1e-3
    model=Sequential()
    model.add(input_layer)
    model.add(rnn)
    model.add(logits)
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # OPTIONAL: Implement
    input_layer=InputLayer(input_shape[1:])
    encoder_RNN=(GRU(64,return_sequences=False))
    repeat_enc_representation = RepeatVector(output_sequence_length)  
    decoder_RNN=(GRU(64,return_sequences=True))
    logits=TimeDistributed(Dense(french_vocab_size,activation='softmax'))
    
    
    learning_rate=1e-3
    model=Sequential()
    model.add(input_layer)
    model.add(encoder_RNN)
    model.add(repeat_enc_representation)
    model.add(decoder_RNN)
    model.add(logits)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
     
    return model

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Building the layers
    embed_layer=Embedding(english_vocab_size,128,input_length=input_shape[1])
    encoder_RNN=Bidirectional(GRU(256,return_sequences=False))
    repeat_enc_representation = RepeatVector(output_sequence_length)  
    decoder_RNN=Bidirectional(GRU(256,return_sequences=True))
    logits=TimeDistributed(Dense(french_vocab_size,activation='softmax'))
    
    
    # TODO: Implement
    learning_rate=0.005
    model=Sequential()
    model.add(embed_layer)
    model.add(encoder_RNN)
    model.add(repeat_enc_representation)
    model.add(decoder_RNN)
    model.add(logits)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model