import os

import tensorflow as tf
import time
import numpy as np
import sys
import json

from tensorflow.keras.layers import Layer, Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Embedding, Dense, Dropout, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model

import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import random

# for logging
from datetime import datetime
import logging


stop = tf.stop_gradient
log1mexp = tfp.math.log1mexp

@tf.function
def log_sigmoid(logits):
    return tf.clip_by_value(tf.math.log_sigmoid(logits), clip_value_max=-1e-7, clip_value_min=-float('inf'))

@tf.function
def logaddexp(x1, x2):
    delta = tf.where(x1 == x2, 0., x1 - x2)
    return tf.math.maximum(x1, x2) + tf.math.softplus(-tf.math.abs(delta))

@tf.function
def log_pr_exactly_k(logp, logq, k):

    batch_size = 40
    n = logp.shape[1]
    
    # state = np.ones((batch_size, k+2)) * -float('inf')
    state = np.ones((batch_size, k+2)) * -300.
    state[:, 1] = 0
    state = tf.convert_to_tensor(state, dtype=tf.float32)

    a = tf.TensorArray(tf.float32, size=n+1)
    a = a.write(0, state)
    
    for i in range(1, n+1):
        
        state = tf.concat([
            # tf.ones([batch_size, 1]) * -float('inf'), 
            tf.ones([batch_size, 1]) * -300., 
            logaddexp(
                state[:, :-1] + logp[:, i-1:i], 
                state[:, 1:] + logq[:, i-1:i]
            )
        ], 1)
        
        a = a.write(i, state)
    a = tf.transpose(a.stack(), perm=[1, 0, 2])
    return a

def marginals(theta, k):
    log_p = log_sigmoid(theta) 
    log_p_complement = log1mexp(log_p) 
    with tf.GradientTape() as tape:
        tape.watch(log_p)
        a = log_pr_exactly_k(log_p, log_p_complement, 10)
        log_pr = a[:, -1, k+1:k+2]
    return tape.gradient(log_pr, log_p), a

@tf.function
def sample(a, probs):
    
    n = a.shape[-2] - 1
    k = a.shape[-1] - 1
    bsz = a.shape[0]

    j = tf.fill((bsz,), k)
    samples = tf.TensorArray(tf.int32, size=n, clear_after_read=False)
    
    for i in tf.range(n, 0, -1):
        
        # Unnormalized probabilities of Xi and -Xi
        full = tf.fill((bsz,), i-1)
        p_idx = tf.stack([full, j-1], axis=1)
        z_idx = tf.stack([full + 1, j], axis=1)
        
        p = tf.gather_nd(batch_dims=1, indices=p_idx, params=a)
        z = tf.gather_nd(batch_dims=1, indices=z_idx, params=a)
        
        p = (p + probs[:, i-1]) - z
        q = log1mexp(p)

        # Sample according to normalized dist.
        X = tfp.distributions.Bernoulli(logits=(p-q)).sample()

        # Pick next state based on value of sample
        j = tf.where(X>0, j - 1, j)

        # Concatenate to samples
        samples = samples.write(i-1, X)
        
    samples = tf.transpose(samples.stack(), perm=[1, 0])    
    return tf.cast(samples, tf.float32)


def gumbel_keys(w):
    # sample some gumbels
    uniform = tf.random.uniform(
        tf.shape(w),
        minval=EPSILON,
        maxval=1.0)
    z = tf.math.log(-tf.math.log(uniform))
    w = w + z
    return w


def continuous_topk(w, k, t, separate=False):
    khot_list = []
    onehot_approx = tf.zeros_like(w, dtype=tf.float32)
    for i in range(k):
        khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
        w += tf.math.log(khot_mask)
        onehot_approx = tf.nn.softmax(w / t, axis=-1)
        khot_list.append(onehot_approx)
    if separate:
        return khot_list
    else:
        return tf.reduce_sum(khot_list, 0)


def sample_subset(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    w = gumbel_keys(w)
    return continuous_topk(w, k, t)

def subset_precision(modelTestInput):
    data = []
    num_annotated_reviews = 0
    with open("data/annotations.json") as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
            num_annotated_reviews = num_annotated_reviews + 1

    selected_word_counter = 0
    correct_selected_counter = 0

    for anotr in range(num_annotated_reviews):
        #print(anotr),
        ranges = data[anotr][str(aspect)] # the aspect id
        text_list = data[anotr]['x']
        #print(ranges)
        review_length = len(text_list)
        #print(text_list)

        list_test = []
        tokenid_list = [word_to_id.get(token, 0) for token in text_list]
        list_test.append(tokenid_list)

        #print(list_test)
        X_test_subset = np.asarray(list_test)
        X_test_subset = sequence.pad_sequences(X_test_subset, maxlen=350)
        #print(X_test_subset)

        prediction = modelTestInput.predict(X_test_subset)
        prediction = tf.squeeze(prediction, -1)
        #print(np.count_nonzero(prediction[0]))

        #print(prediction[0])
        x_val_selected = prediction[0] * X_test_subset
        #print(tf.cast(x_val_selected, tf.int32))

        selected_words = np.vectorize(id_to_word.get)(x_val_selected)[0][-review_length:]
        selected_nonpadding_word_counter = 0
        
        for i, w in enumerate(selected_words):
            if w != '<PAD>': # we are nice to the L2X approach by only considering selected non-pad tokens
                selected_nonpadding_word_counter = selected_nonpadding_word_counter + 1
                for r in ranges:
                    rl = list(r)
                    if i in range(rl[0], rl[1]):
                        correct_selected_counter = correct_selected_counter + 1
        # we make sure that we select at least 10 non-padding words
        # if we have more than select_k non-padding words selected, we allow it but count that in
        selected_word_counter = selected_word_counter + max(selected_nonpadding_word_counter, select_k)

    return correct_selected_counter / selected_word_counter


class Concatenate(Layer):
    """
    Layer for concatenation. 
    
    """
    def __init__(self, **kwargs): 
        super(Concatenate, self).__init__(**kwargs)

    def call(self, inputs):
        input1, input2 = inputs  
        input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        return tf.concat([input1, input2], axis = -1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)


def construct_gumbel_selector(X_ph, num_words, embedding_dims, maxlen):
    """
    Build the L2X model for selecting words. 

    """
    emb_layer = Embedding(num_words, embedding_dims, weights=[embedding_matrix], input_length=maxlen, trainable=False, name='emb_gumbel')
    emb = emb_layer(X_ph) #(350, 200) 
    #net = Dropout(0.2, name = 'dropout_gumbel')(emb)# this is not used in the L2X experiments
    net = emb
    first_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(net)    

    # global info
    net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer)
    global_info = Dense(100, name = 'new_dense_1', activation='relu')(net_new) 

    # local info
    net = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(first_layer) 
    local_info = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(net)  
    combined = Concatenate()([global_info,local_info]) 
    net = Dropout(0.2, name = 'new_dropout_2')(combined)
    net = Conv1D(100, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)   

    logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net)  
    
    return logits_T

class IMLESubsetkLayer(tf.keras.layers.Layer):
  
    def __init__(self, _k=10, _tau=1.0, _lambda=1.0):
        super(IMLESubsetkLayer, self).__init__()
        
        self.k = _k
        self._tau = _tau
        self._lambda = _lambda
        self.samples = None
    
    def imle_layer(self, logits, hard=False):
        logp = log_sigmoid(logits)
        logq = log1mexp(logp)

        a = log_pr_exactly_k(logp, logq, self.k)
        samples_p = sample(a, logp)
        y = marginals(logits, self.k)[0]
        return tf.stop_gradient(samples_p - y) + y

    @tf.custom_gradient
    def imle_layer_clip(self, logits, hard=False):
        # clipping for numerical stability
        logp = log_sigmoid(logits)
        logq = log1mexp(logp)

        a = log_pr_exactly_k(logp, logq, self.k)
        samples_p = sample(a, logp)
        y = 0.99 * marginals(logits, self.k)[0] + 0.01 * tf.math.sigmoid(logits)
        y = tf.clip_by_value(y, clip_value_max=1-1e-3, clip_value_min=1e-3)
        
        def custom_grad(dy):
            return tf.clip_by_norm(dy, 0.5), hard

        return tf.stop_gradient(samples_p - y) + y, custom_grad
    
    def call(self, logits, hard=False, clip=False):
        logits = tf.squeeze(logits, -1)
        if not clip:
            y = self.imle_layer(logits, hard)
        else:
            y = self.imle_layer_clip(logits, hard)
        return tf.expand_dims(y, -1)

    def get_config(self):
        cfg = super().get_config()
        return cfg


class Model_MSE(tf.keras.Model):
    def __init__(self, model, k, name=None):
        super(Model_MSE, self).__init__(name=name)
        self.model = model
        self.k = k

    def call(self, x):
        
        global_info_layers = ['emb_gumbel', 'conv1_gumbel', 'new_global_max_pooling1d_1', 'new_dense_1']
        local_info_layers = ['emb_gumbel', 'conv1_gumbel', 'conv2_gumbel', 'conv3_gumbel']

        x = model.get_layer('input_1')(x)

        global_info = x
        for layer_name in global_info_layers:
            global_info = model.get_layer(layer_name)(global_info)
            
        local_info = x
        for layer_name in local_info_layers:
            local_info = model.get_layer(layer_name)(local_info)
            
        combined = model.get_layer('concatenate')([global_info, local_info])
        
        logits_T_layers = ['new_dropout_2', 'conv_last_gumbel', 'conv4_gumbel']        
        logits_T = combined
        for layer_name in logits_T_layers:
            logits_T = model.get_layer(layer_name)(logits_T)
        
        # MAP from IMLESubsetkLayer
        logits = logits_T
        logits = tf.squeeze(logits, -1)
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:,-1], -1)
        z_test = tf.cast(tf.greater_equal(logits, threshold), tf.float32)        
        T = tf.expand_dims(z_test, -1)
        
        emb2 = model.get_layer("embedding")(x)
        net = model.get_layer("lambda")(model.get_layer("multiply")([emb2, T]))
        
        net_layers = ['dense', 'activation', 'new_dense']
        for layer_name in net_layers:
            net = model.get_layer(layer_name)(net)
            
        return net


class Model_Precision(tf.keras.Model):
    def __init__(self, model, k, name=None):
        super(Model_Precision, self).__init__(name=name)
        self.model = model
        self.k = k

    def call(self, x):
        
        global_info_layers = ['emb_gumbel', 'conv1_gumbel', 'new_global_max_pooling1d_1', 'new_dense_1']
        local_info_layers = ['emb_gumbel', 'conv1_gumbel', 'conv2_gumbel', 'conv3_gumbel']

        x = model.get_layer('input_1')(x)

        global_info = x
        for layer_name in global_info_layers:
            global_info = model.get_layer(layer_name)(global_info)
            
        local_info = x
        for layer_name in local_info_layers:
            local_info = model.get_layer(layer_name)(local_info)
            
        combined = model.get_layer('concatenate')([global_info, local_info])
        
        logits_T_layers = ['new_dropout_2', 'conv_last_gumbel', 'conv4_gumbel']        
        logits_T = combined
        for layer_name in logits_T_layers:
            logits_T = model.get_layer(layer_name)(logits_T)
        
        # MAP from IMLESubsetkLayer
        logits = logits_T
        logits = tf.squeeze(logits, -1)
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:,-1], -1)
        z_test = tf.cast(tf.greater_equal(logits, threshold), tf.float32)        
        T = tf.expand_dims(z_test, -1)
                    
        return T


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lam", type=float, default=1000.)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aspect", type=int, default=1)
    # parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--clip", action='store_true')
    parser.add_argument("--n_epochs",
    type=int, default=20)
    parser.add_argument("--k", type=int, default=10)

    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)


    """
    logging
    """
    log_name = f"aspect_{args.aspect}_k_{args.k}_lam_{args.lam}_lr_{args.lr}_seed_{args.seed}"
    log_path = os.path.join('log/', log_name)
    os.mkdir(log_path)
    # logging.basicConfig(
    #     filename=log_path + f'/{log_name}.log', level=logging.INFO
    # )


    """
    set random seed
    """
    seed = args.seed    
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'
    os.environ['PYTHONHASHSEED'] = f'{seed}'
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # logging.info(f"random seed = {seed}")


    """
    loading data
    """
    aspect = args.aspect
    input_path_train = "data/reviews.aspect" + str(aspect) + ".train.txt"
    input_path_validation = "data/reviews.aspect" + str(aspect) + ".heldout.txt"

    # the dictionary mapping words to their IDs
    word_to_id = dict()
    token_id_counter = 3


    with open(input_path_train) as fin:
        for line in fin:
            y, sep, text = line.partition("\t")
            token_list = text.split(" ")
            for token in token_list:
                if token not in word_to_id:
                    word_to_id[token] = token_id_counter
                    token_id_counter = token_id_counter + 1
            
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value:key for key,value in word_to_id.items()}

    X_train_list = []
    Y_train_list = []
    # now we iterate again to assign IDs
    with open(input_path_train) as fin:
        for line in fin:
            y, sep, text = line.partition("\t")
            token_list = text.split(" ")
            tokenid_list = [word_to_id[token] for token in token_list]
            X_train_list.append(tokenid_list)
            
            # extract the normalized [0,1] value for the aspect
            y = [ float(v) for v in y.split() ]
            Y_train_list.append(y[aspect])

    #print(y_list)        
    X_train = np.asarray(X_train_list)
    Y_train = np.asarray(Y_train_list)

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=350)    

    print("Loading heldout data...")
    X_val_list = []
    Y_val_list = []
    # now we iterate again to assign IDs
    with open(input_path_validation) as fin:
        for line in fin:
            y, sep, text = line.partition("\t")
            token_list = text.split(" ")
            tokenid_list = [word_to_id.get(token, 2) for token in token_list]
            X_val_list.append(tokenid_list)
            
            # extract the normalized [0,1] value for the aspect
            y = [ float(v) for v in y.split() ]
            Y_val_list.append(y[aspect])

    #print(y_list)        
    X_val_both = np.asarray(X_val_list)
    Y_val_both = np.asarray(Y_val_list)

    print('Pad sequences (samples x time)')
    X_val_both = sequence.pad_sequences(X_val_both, maxlen=350)

    print(X_train.shape)


    """
    set parameters
    """
    max_features = token_id_counter+1
    maxlen = 350
    batch_size = 40
    embedding_dims = 200
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    # epochs = 20
    # select_k = 10 # Number of selected words by the methods
    select_k = args.k


    """
    load work embeddings
    """
    # this cell loads the word embeddings from the external data
    embeddings_index = {}
    f = open("data/review+wiki.filtered.200.txt")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_to_id) + 1, 200))
    for word, i in word_to_id.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    """
    model training
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.batch(batch_size)
    
    print('Creating model...')
        
    # create a new validation/test split
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_both, Y_val_both, test_size=0.5, random_state=0)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.batch(batch_size)
    
    # P(S|X)
    X_ph = Input(shape=(maxlen,), dtype='int32')    

    logits_T = construct_gumbel_selector(X_ph, max_features, embedding_dims, maxlen)
    
    subsetklayer_tau = 2.0
    subsetklayer_lambda = args.lam
    # SIMPLE layer
    T = IMLESubsetkLayer(select_k, _tau=subsetklayer_tau, _lambda=subsetklayer_lambda)(logits_T, hard=False, clip=args.clip)
        
    # q(X_S)
    # Define various Keras layers.
    Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(select_k), output_shape=lambda x: [x[0],x[2]])
    emb2 = Embedding(max_features, embedding_dims, input_length=maxlen, weights=[embedding_matrix], trainable=False)(X_ph)
    net = Mean(Multiply()([emb2, T]))
    net = Dense(hidden_dims)(net)
    net = Activation('relu')(net)
    preds = Dense(1, activation='sigmoid', name = 'new_dense')(net)

    model = Model(inputs=X_ph, outputs=preds)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(loss=['mse'], optimizer=optimizer, metrics=['mse'])

    filepath = f"{log_path}/l2x-exact.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint] 
    st = time.time()

    n_epochs = args.n_epochs

    model.fit(train_dataset, validation_data=val_dataset, callbacks=callbacks_list, epochs=n_epochs)
    duration = time.time() - st
    print('Training time is {}'.format(duration))
    # logging.info('Training time is {}'.format(duration))


    """
    model evaluation: MSE, precision by MAP
    """
    # MSE
    model_mse = Model_MSE(model, select_k)
    model_mse.compile(loss=['mse'], optimizer=optimizer, metrics=['mse'])
    
    results = model_mse.evaluate(X_test, Y_test, batch_size=batch_size)
    print(f"test loss, test acc: {results}")    

    # precision
    model_precision = Model_Precision(model, select_k)
    model_precision.compile(loss=['mse'], optimizer=optimizer, metrics=['mse'])
    
    subset_prec = subset_precision(model_precision)
    print(f"Subset precision: {subset_prec}")