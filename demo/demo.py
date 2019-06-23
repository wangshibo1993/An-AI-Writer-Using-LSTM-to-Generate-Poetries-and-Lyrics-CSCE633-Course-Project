import keras
from keras.models import load_model
from scipy import misc, spatial
from PIL import Image
import numpy as np
import os
import tkinter as tk
from tkinter import *

import tensorflow as tf
import numpy as np
import pickle
from keras.models import load_model
import json

with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char] = pickle.load(fr)


batch_size = 1
hidden_size = 256
num_layer = 2
embedding_size = 256

X = tf.placeholder(tf.int32, [batch_size, None])
Y = tf.placeholder(tf.int32, [batch_size, None])
learning_rate = tf.Variable(0.0, trainable=False)

cell = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True) for i in range(num_layer)], 
    state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)

embeddings = tf.Variable(tf.random_uniform([len(char2id) + 1, embedding_size], -1.0, 1.0))
embedded = tf.nn.embedding_lookup(embeddings, X)

outputs, last_states = tf.nn.dynamic_rnn(cell, embedded, initial_state=initial_state)


outputs = tf.reshape(outputs, [-1, hidden_size])
logits = tf.layers.dense(outputs, units=len(char2id) + 1)
probs = tf.nn.softmax(logits)
targets = tf.reshape(Y, [-1])

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
params = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5)
optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, params))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./50'))

def poetry_generate():
    states_ = sess.run(initial_state)
    
    gen = ''
    c = '['
    while c != ']':
        gen += c
        x = np.zeros((batch_size, 1))
        x[:, 0] = char2id[c]
        probs_, states_ = sess.run([probs, last_states], feed_dict={X: x, initial_state: states_})
        probs_ = np.squeeze(probs_)
        pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))
        while pos not in id2char:
            pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))
        c = id2char[pos]
    
    return gen[1:]
    #t.delete('1.0','end')
    #t.insert("insert", gen[1:])

def poetry_generate_with_head(head):
    states_ = sess.run(initial_state)
    
    gen = ''
    c = '['
    i = 0
    while c != ']':
        gen += c
        x = np.zeros((batch_size, 1))
        x[:, 0] = char2id[c]
        probs_, states_ = sess.run([probs, last_states], feed_dict={X: x, initial_state: states_})
        probs_ = np.squeeze(probs_)
        pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))

        if (c == '[' or c == '。' or c == '，') and i < len(head):
            c = head[i]
            i += 1
        else:
            while pos not in id2char:
                pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))
            c = id2char[pos]
    
    return gen[1:]



def poetry_generate_with_topic(head):
    states_ = sess.run(initial_state)
    
    gen = ''
    c = '['
    i = 0
    
    while c != ']':
        
        gen += c
        
        x = np.zeros((batch_size, 1))
        x[:, 0] = char2id[c]
        probs_, states_ = sess.run([probs, last_states], feed_dict={X: x, initial_state: states_})
        probs_ = np.squeeze(probs_)
        pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))
        while pos not in id2char:
            pos = int(np.searchsorted(np.cumsum(probs_), np.random.rand() * np.sum(probs_)))
        c = id2char[pos]
        if i<len(head):
            c=head[i]
            i+=1
            
    
    
    
    return gen[1:]







def loadParam(model_file, word2index_file, index2word_file):
    """
    load model and word2index_file, index2word_file
    :param model_file:
    :param word2index_file:
    :param index2word_file:
    :return:
    """
    # get model.
    model = load_model(model_file)
    # get the word2index and index2word data.
    with open(word2index_file, 'r', encoding='utf8') as f:
        json_obj = f.read()
        word2index = json.loads(json_obj)
        f.close()
    with open(index2word_file, 'r', encoding='utf8') as f:
        json_obj = f.read()
        index2word = json.loads(json_obj)
        f.close()
    index2word_new = {}
    for key, value in index2word.items():
        index2word_new[int(key)] = value
    return model, word2index, index2word_new

def sample(preds, diversity = 1.0):
    """
    get the max probability index.
    :param preds: prediction
    :param diversity:
    :return:
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def lyrics_generate(start, model, word2index, index2word, SEQ_LENGTH, generate_maxlen):
    """
    generate lyrics according start sentence.
    :param start: startWith sentence
    :param model:
    :param word2index:
    :param index2word:
    :param maxlen: the length of generating sentence.
    :return:
    """
    sentence = start[:SEQ_LENGTH]   
    diversity = 1.0
    while len(sentence) < generate_maxlen:
        
        x_pred = np.zeros((1, SEQ_LENGTH))    

        min_index = max(0, len(sentence) - SEQ_LENGTH)    
        for idx in range(min_index, len(sentence)):
            x_pred[0, SEQ_LENGTH - len(sentence) + idx] = word2index.get(sentence[idx], 1)  

        preds = model.predict(x_pred, verbose=0)[0]   
        next_index = sample(preds, diversity)   
        next_word = index2word[next_index]
        if not (next_word == '。' and sentence[-1] == '。'):   
            sentence = sentence + next_word  
    return sentence



model_file = './model_epoch50_2lstm_1dense_seq50_phrase_based_best.h5'
word2index_file = './word_to_index_word.txt'
index2word_file = './index_to_word_word.txt'
model, word2index, index2word = loadParam(model_file, word2index_file, index2word_file)
generate_maxlen = 200
SEQ_LENGTH = 3















window = tk.Tk()
window.title('RNN Generating Poetry')
window.geometry('400x400')
# e = tk.Entry(window, show="*")
e = tk.Entry(window)
e.pack()

def insert_point1():
    var = e.get()
    if var=="":
        poetry=poetry_generate()
    else:
        poetry=poetry_generate_with_head(var)
    t.delete('1.0','end')
    t.insert("insert", poetry)


def insert_point2():
    var = e.get()
    if var=="":
        poetry=poetry_generate()
    else:
        poetry=poetry_generate_with_topic(var)
    t.delete('1.0','end')
    t.insert("insert", poetry)
    
def insert_point3():
    var = e.get()
    Lyrics=lyrics_generate(var, model, word2index, index2word, SEQ_LENGTH, generate_maxlen)
    Lyrics = Lyrics.replace("。", '\n')
    t.delete('1.0','end')
    t.insert("insert", Lyrics)
    
#b1 = tk.Button(window, text='Generate poetry Randomly', width=25,height=2, command=insert_point1)
#b1.pack()
b2 = tk.Button(window, text='Generate Chinese poetry with Head', width=33,height=3, command=insert_point1)
b2.pack()
b3 = tk.Button(window, text='Generate Chinese poetry with Topic', width=33,height=3, command=insert_point2)
b3.pack()
b4 = tk.Button(window, text='Generate Chinese Lyrics with Topic', width=33,height=3, command=insert_point3)
b4.pack()



t = tk.Text(window, height=6)
t.pack()

window.mainloop()
