"""
Use this script to train the sonnet writing with keras using LSTM
TODO: optimize plan 
1. fit tokenizer right after text cleanning
2. add more sonnet work to author similar to shakespear

Note that there seems to be some compatibility issues with keras model, swtch to tensorflow.keras resolve them somehow 
"""
import spacy
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras import Sequential
#from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
#from keras.layers import Dense, LSTM, Embedding
from pickle import dump


"""define useful functions"""

def read_file(filepath):
    with open(filepath) as file:
        str_text = file.read()

        return str_text


def separate_punc(doc_text):
    
    return [token.text.strip().lower() 
            for token in nlp(doc_text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer","ner"]) 
            if token.text not in '1234567890!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\n\n\s']


def gen_sequence(seq_len, tokens):
    # generate text_sequences from a token list
    train_len = 10 + 1
    text_sequences = []

    for i in range(train_len, len(tokens)):
        seq = tokens[i - train_len:i]
        text_sequences.append(seq)

    return text_sequences


def text_to_numercial(text_sequences):
    # format sequences of texts to sequences of numeric values that keras can understand
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)

    # do an inventory of the vocabulary size
    vocab_size = len(tokenizer.word_counts)
    print(f"The vocabulary size of the corpus to be fittes is {vocab_size}.")

    numerical_sequences = tokenizer.texts_to_sequences(text_sequences)
    print("Saving the tokenizer in pickele ...")
    dump(tokenizer, open('sonnet_tokenizer', 'wb'))

    return (numerical_sequences, vocab_size)


def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    model.summary()
    
    return model


if __name__ == "__main__":
    """preprocessing"""
    print("Reading the sonnet text data...")
    sonnet = read_file('sonnet.txt')

    # load spacy en model and use it to tokenize
    print("Tokenization...")
    nlp = spacy.load('en_core_web_sm')
    #disable=['parser', 'tagger', 'ner'])
    #nlp.max_length = 98331

    tokens = separate_punc(sonnet)
    tokens = list(filter(lambda x: x != '', tokens))
    tokens.remove("'s")

    print("Transforming text data to sequences...")
    text_sequences = gen_sequence(seq_len=10+1, tokens=tokens)
    numerical_sequences, vocab_size = text_to_numercial(text_sequences)
    numerical_sequences = np.array(numerical_sequences)
    print(f"The sequences have shape of {numerical_sequences.shape}")

    """training"""
    # grabbing the sequences except for the last element of each sequence
    
    X = numerical_sequences[:, :-1]
    y = numerical_sequences[:, -1]

    # the predicted outcome is the possibities of the whole vocabuary
    y = to_categorical(y, num_classes=vocab_size + 1)

    # seq_len is the length of a general sentence, and the length of a sentence is usually no more than 10 words
    model = create_model(vocabulary_size=vocab_size + 1, seq_len=10)
    model.fit(X, y, batch_size=128, epochs=200, verbose=2)

    """finally save the model and the tokenizer"""
    model.save('sonnet_writer_v2.h5')
