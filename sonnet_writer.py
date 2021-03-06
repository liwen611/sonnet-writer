#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 21:24:57 2021

@author: liwenhuang

Note that model is trained on sequential length of 10 as a sonnect usually have 10 syllables a line
"""
from random import randint
from pickle import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# supress the tf annoying warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    '''
    INPUTS:
    model : model that was trained on text data
    tokenizer : tokenizer that was fit on text data
    seq_len : length of training sequence
    seed_text : raw string text to serve as the seed
    num_gen_words : number of words to be generated by model
    '''
    
    # Final Output
    output_text = []
    
    # Intial Seed Sequence
    input_text = seed_text
    
    # Create num_gen_words
    for i in range(num_gen_words):
        
        # Take the input text string and encode it to a sequence
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        
        # Pad sequences to our trained rate (50 words in the video)
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        
        # Predict Class Probabilities for each word
        #pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        pred_word_ind = np.argmax(model.predict(pad_encoded), axis=-1)[0]
        
        # Grab word
        pred_word = tokenizer.index_word[pred_word_ind] 
        
        # Update the sequence of input text (shifting one over with the new word)
        input_text += ' ' + pred_word
        
        output_text.append(pred_word)
        
    # Make it look like a sentence.
    return ' '.join(output_text)

if __name__ == "__main__":
    tokenizer = load(open('./models/sonnet_tokenizer', 'rb'))
    model = load_model('./models/sonnet_writer_v1.h5')
    print('Type a few words to begins your sonnet:')
    seed_text = input()
    need_len = 10 - len(seed_text.split())
    gen_text = generate_text(model, tokenizer, 10, seed_text, need_len)
    print("...")
    print(seed_text + " " + gen_text)
    
    while input("Continue writing? [y/n]") == "y":
        seed_text = input()
        need_len = 10 - len(seed_text.split())
        gen_text = generate_text(model, tokenizer, 10, seed_text, need_len)
        print("...")
        print(seed_text + " " + gen_text)
    # do something
    print("End of your master piece!")