import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # get P-T points
    X = [series[i:i+window_size] for i in range(len(series)-window_size)]
    y = [series[i] for i in range(window_size, len(series))] 

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(7, 1)))
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    to_remove = ['-', '\'','(', ')', "\"", '&', 'é', '/', '*', '@', '$' ]
    # show all uniques char to further filter
    unique = {}
    for ch in text:
        if ch in unique:
            unique[ch] += 1
        else:
            unique[ch] = 0
    
    unique = dict([(k, v) for k,v in unique.items() if v != 0]) 
    print(unique)
    
    tmp_text = list(text)
    for ch_idx, ch in enumerate(tmp_text):
        if ch not in punctuation and ch in to_remove:
            tmp_text[ch_idx] = ''

    text = ''.join(tmp_text)
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    inputs = [text[i:i+window_size] for i in range(0, len(text)-window_size, step_size)]
    outputs = [text[i] for i in range(window_size, len(text)-window_size, step_size)]
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
