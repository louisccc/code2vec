from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import tensorflow as tf


class code2vec(tf.keras.Model): 

    def __init__(self, config):
        super(code2vec, self).__init__()

        self.config = config 

        self.def_parameters()

    def def_parameters(self):        
        emb_initializer = tf.initializers.glorot_normal()
        self.ents_embeddings = tf.Variable(emb_initializer(shape=(self.config.num_of_words, self.config.embedding_size)), name='ents')
        self.path_embeddings = tf.Variable(emb_initializer(shape=(self.config.num_of_paths, self.config.embedding_size)), name='paths')
        self.tags_embeddings = tf.Variable(emb_initializer(shape=(self.config.num_of_tags, self.config.code_embedding_size)), name='tags')
        self.attention_param = tf.Variable(emb_initializer(shape=(self.config.code_embedding_size, 1)), name='attention_param')
        self.transform_matrix= tf.Variable(emb_initializer(shape=(3*self.config.embedding_size, self.config.code_embedding_size)), name='transform')

    def forward(self, e1, p, e2, train=True):
        # e1_e is [batch_size, max_contexts, embeddings size]
        # p_e  is [batch_size, max_contexts, embeddings size]
        # e2_e is [batch_size, max_contexts, embeddings size]
        e1_e = tf.nn.embedding_lookup(params=self.ents_embeddings, ids=e1)
        p_e  = tf.nn.embedding_lookup(params=self.path_embeddings, ids=p)
        e2_e = tf.nn.embedding_lookup(params=self.ents_embeddings, ids=e2)

        # context_emb = [batch_size, max_contexts, 3*embedding_size]        
        context_e = tf.concat([e1_e, p_e, e2_e], axis=-1) 

        # apply a dropout to context emb. 
        if train:
            context_e = tf.nn.dropout(context_e, rate=1-self.config.dropout_factor)

        # flatten context embeddings => [batch_size*max_contexts, 3*embedding_size]
        context_e = tf.reshape(context_e, [-1, 3*self.config.embedding_size])

        # tranform context embeddings -> to [batch_size*max_contexts, code_embedding_size]
        flat_emb = tf.tanh(tf.matmul(context_e, self.transform_matrix))

        # calculate weights => to [batch_size*max_contexts, 1]
        contexts_weights = tf.matmul(flat_emb, self.attention_param)

        # reshapeing context weights => to [batch_size, max_contexts, 1]
        batched_contexts_weights = tf.reshape(contexts_weights, [-1, self.config.max_contexts, 1])

        # calculate softmax for attention weights. 
        attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)

        # reshaping the embeddings => to [batch_size, max_contexts, code_embedding_size]
        batched_flat_emb = tf.reshape(flat_emb, [-1, self.config.max_contexts, self.config.code_embedding_size])

        # calculating the code vectors => to [batch_size, code_embedding_size]
        code_vectors = tf.reduce_sum(tf.multiply(batched_flat_emb, attention_weights), axis=1)

        return code_vectors, attention_weights