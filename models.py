from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.layers.merge import concatenate

from custom_layers import Attention

import variables

import h5py
import os

def save_model_without_embedding(model, savefile):
    model.save_weights('models/temp.h5')
    with h5py.File('models/temp.h5', 'r+') as fs, h5py.File(savefile, 'w') as fd:
        fd.attrs['layer_names'] = [l.name.encode('utf8') for l in model.layers if l.name != 'embedding']
        fd.attrs['backend'] = fs.attrs['backend']
        fd.attrs['keras_version'] = fs.attrs['keras_version']
        for key in fs.keys():
            if key != u'embedding':
                fs.copy(key, fd)
        fd.flush()
    os.remove('models/temp.h5')

def cbow(data, embedding_weights, verbose=True):
    dropout = 0.6
    maxlen = variables.MAX_LEN

    nb_words, embedding_dim = embedding_weights.shape
    
    # Inputs
    text = Input(shape=(embedding_dim,), name='data')
    titl = Input(shape=(embedding_dim,), name='title')
    ratg = Input(shape=(1,), name='rating')
    
    # Merge
    m = concatenate([text, titl, ratg])
    
    #Output
    s = Dropout(dropout)(m)
    s = Dense(512, name='dense', activation='relu')(s)
    s = Dropout(dropout)(s)
    s = Dense(1, name='preds', activation='sigmoid')(s)
    
    model = Model(inputs=[text,titl,ratg],outputs=[s])
    
    if verbose:
        model.summary()
    
    data_train = ([data['train']['text_vec'], 
                   data['train']['titl_vec'], 
                   data['train']['ratg']], 
                  data['train']['y'])
    data_val =   ([data['val']['text_vec'], 
                   data['val']['titl_vec'], 
                   data['val']['ratg']], 
                  data['val']['y'])
    data_test =  [data['test']['text_vec'], 
                  data['test']['titl_vec'], 
                  data['test']['ratg']]
            
    return model, data_train, data_val, data_test

def lstm(data, embedding_weights, verbose=True):
    lstm_output = 64
    lstm_dropout_w = 0.3
    lstm_dropout_u = 0.3
    dropout = 0.6
    maxlen = variables.MAX_LEN

    nb_words, embedding_dim = embedding_weights.shape
    
    embedding = Embedding(input_dim=nb_words,             
                          input_length=maxlen,            
                          output_dim=embedding_dim,       
                          mask_zero=True,                 
                          weights=[embedding_weights],    
                          trainable=False,
                          name='embedding')
    lstm = LSTM(lstm_output,                         
                dropout=lstm_dropout_u,            
                recurrent_dropout=lstm_dropout_w,
                return_sequences=False,
                name='lstm')
    
    # Inputs
    text = Input(shape=(maxlen,), name='data')
    titl = Input(shape=(maxlen,), name='title')
    ratg = Input(shape=(1,), name='rating')
    
    # Embedding
    text_embedding = embedding(text)
    titl_embedding = embedding(titl)
    
    # LSTM
    text_lstm = lstm(text_embedding)
    titl_lstm = lstm(titl_embedding)
    
    # Merge
    m = concatenate([text_lstm, titl_lstm, ratg])
    
    #Output
    s = Dropout(dropout)(m)
    s = Dense(32, name='dense', activation='relu')(s)
    s = Dropout(dropout)(s)
    s = Dense(1, name='preds', activation='sigmoid')(s)
    
    model = Model(inputs=[text,titl,ratg],outputs=[s])
    
    if verbose:
        model.summary()
    
    data_train = ([data['train']['text_ids'], 
                   data['train']['titl_ids'], 
                   data['train']['ratg']], 
                  data['train']['y'])
    data_val =   ([data['val']['text_ids'], 
                   data['val']['titl_ids'], 
                   data['val']['ratg']], 
                  data['val']['y'])
    data_test =  [data['test']['text_ids'], 
                  data['test']['titl_ids'], 
                  data['test']['ratg']]
            
    return model, data_train, data_val, data_test

def bilstm(data, embedding_weights, verbose=True):
    lstm_output = 32
    lstm_dropout_w = 0.3
    lstm_dropout_u = 0.3
    dropout = 0.6
    maxlen = variables.MAX_LEN

    nb_words, embedding_dim = embedding_weights.shape
    
    embedding = Embedding(input_dim=nb_words,             
                          input_length=maxlen,            
                          output_dim=embedding_dim,       
                          mask_zero=True,                 
                          weights=[embedding_weights],    
                          trainable=False,
                          name='embedding')
    lstm = Bidirectional(LSTM(lstm_output,                         
                dropout=lstm_dropout_u,            
                recurrent_dropout=lstm_dropout_w,
                return_sequences=False), name='lstm')
    
    # Inputs
    text = Input(shape=(maxlen,), name='data')
    titl = Input(shape=(maxlen,), name='title')
    ratg = Input(shape=(1,), name='rating')
    
    # Embedding
    text_embedding = embedding(text)
    titl_embedding = embedding(titl)
    
    # LSTM
    text_lstm = lstm(text_embedding)
    titl_lstm = lstm(titl_embedding)
    
    # Merge
    m = concatenate([text_lstm, titl_lstm, ratg])
    
    #Output
    s = Dropout(dropout)(m)
    s = Dense(32, name='dense', activation='relu')(s)
    s = Dropout(dropout)(s)
    s = Dense(1, name='preds', activation='sigmoid')(s)
    
    model = Model(inputs=[text,titl,ratg],outputs=[s])
    
    if verbose:
        model.summary()
    
    data_train = ([data['train']['text_ids'], 
                   data['train']['titl_ids'], 
                   data['train']['ratg']], 
                  data['train']['y'])
    data_val =   ([data['val']['text_ids'], 
                   data['val']['titl_ids'], 
                   data['val']['ratg']], 
                  data['val']['y'])
    data_test =  [data['test']['text_ids'], 
                  data['test']['titl_ids'], 
                  data['test']['ratg']]
            
    return model, data_train, data_val, data_test

def attention_lstm(data, embedding_weights, verbose=True):
    lstm_output = 64
    lstm_dropout_w = 0.3
    lstm_dropout_u = 0.3
    dropout = 0.6
    maxlen = variables.MAX_LEN

    nb_words, embedding_dim = embedding_weights.shape
    
    embedding = Embedding(input_dim=nb_words,             
                          input_length=maxlen,            
                          output_dim=embedding_dim,       
                          mask_zero=True,                 
                          weights=[embedding_weights],    
                          trainable=False,
                          name='embedding')
    lstm = LSTM(lstm_output,                         
                dropout=lstm_dropout_u,            
                recurrent_dropout=lstm_dropout_w,
                return_sequences=True,
                name='lstm')
    attention = Attention()
    
    # Inputs
    text = Input(shape=(maxlen,), name='data')
    titl = Input(shape=(embedding_dim,), name='title')
    ratg = Input(shape=(1,), name='rating')
    
    # Embedding
    text_embedding = embedding(text)
    
    # LSTM
    text_lstm = attention(lstm(text_embedding))
    
    # Projection
    titl_p = Dropout(dropout)(titl)
    titl_p = Dense(lstm_output, name='proj', activation='relu')(titl_p)
    
    # Merge
    m = concatenate([text_lstm, titl_p, ratg])
    
    #Output
    s = Dropout(dropout)(m)
    s = Dense(32, name='dense', activation='relu')(s)
    s = Dropout(dropout)(s)
    s = Dense(1, name='preds', activation='sigmoid')(s)
    
    model = Model(inputs=[text,titl,ratg],outputs=[s])
    
    if verbose:
        model.summary()
    
    data_train = ([data['train']['text_ids'], 
                   data['train']['titl_vec'], 
                   data['train']['ratg']], 
                  data['train']['y'])
    data_val =   ([data['val']['text_ids'], 
                   data['val']['titl_vec'], 
                   data['val']['ratg']], 
                  data['val']['y'])
    data_test =  [data['test']['text_ids'], 
                  data['test']['titl_vec'], 
                  data['test']['ratg']]
            
    return model, data_train, data_val, data_test