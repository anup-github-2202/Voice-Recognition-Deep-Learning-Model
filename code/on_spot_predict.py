import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import pandas as pd
import librosa
import pickle
import os
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model

def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10),min_periods=1,center=True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def input_audio():
    sample_freq =48000
    duration = 1.5
    print("say")
    array = sd.rec(int(duration*sample_freq),sample_freq,1,blocking = True)
    sd.play(array,sample_freq)
    write('output.wav', sample_freq, array)

def build_predictions(file):
    y_pred = []
    fn_prob = {}
    count_if = 0
    count_elif = 0
    print('Extracting')
    rate, signal = wavfile.read(file)
    y_prob = []
            
    for i in range(0,signal.shape[0]-conv.step,conv.step):
        sample = signal[i:i+conv.step]
        x_sample = mfcc(sample,rate,numcep=conv.nfeat,nfilt=conv.nfilt,nfft=conv.nfft)
        if(x_sample.shape[0]<9):
            x_sample = np.concatenate((x_sample,np.zeros(shape=(9-x_sample.shape[0],20))))
            x = x_sample
            count_if+=1
        elif(x_sample.shape[0]>9):
            x_sample1 = x_sample[0:9]
            x_sample1 = (x_sample1-conv.mean)/conv.std
            x_sample1 = np.expand_dims(x_sample1,axis=0)
            y_hat = model.predict(x_sample1)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            x_sample2 = x_sample[9:x_sample.shape[0]]
            x_sample2 = np.concatenate((x_sample2,np.zeros(shape=(18-x_sample.shape[0],20))))
            x = x_sample2
            count_elif+=1
        else:
            x = x_sample
        x = (x-conv.mean)/conv.std
        x = np.expand_dims(x,axis=0)
        y_hat = model.predict(x)
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))
    print(count_if,count_elif)
    fn_prob[file] = np.mean(y_prob,axis=0).flatten()
            
    return y_pred,fn_prob
            

input_audio()
    
signal, rate = librosa.load('output.wav',sr=16000)
mask = envelope(signal,rate,0.005)
write(filename='output_processed.wav',rate=16000,data=signal[mask])   
    

df = pd.read_csv('C:/Users/98viv/Audio_files_test.csv')
classes = list(np.unique(df.label))
pickle_path = os.path.join('pickle','rnn4.p')

with open(pickle_path,'rb') as handle:
    conv = pickle.load(handle)
    
model = load_model(conv.model_path)

y_pred,fn_prob = build_predictions('output_processed.wav')
print(y_pred)
print(fn_prob)

y_probs = []
y_prob = fn_prob['output_processed.wav']
y_probs.append(y_prob)

y_pred = [classes[np.argmax(y)] for y in y_probs]
print(y_pred)
