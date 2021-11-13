import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score


def build_predictions(audio_dir):
    y_true = []
    y_pred = []
    fn_prob = {}
    
    print('Extracting')
    for folder in tqdm(os.listdir(audio_dir)):
        filenames = os.listdir(os.path.join(audio_dir,folder))
        path = os.path.join(audio_dir,folder)
        for file in tqdm(filenames):
            rate, signal = wavfile.read(os.path.join(path,file))
            label = fn2class[file]
            c = classes.index(label)
            y_prob = []
            x_sample = mfcc(signal,rate,numcep=conv.nfeat,nfilt=conv.nfilt,nfft=conv.nfft)
            if(x_sample.shape[0]<9):
                x_sample = np.concatenate((x_sample,np.zeros(shape=(9-x_sample.shape[0],20))))
                x = x_sample
            elif(x_sample.shape[0]>9):
                x_sample1 = x_sample[0:9]
                x_sample1 = (x_sample1-conv.mean)/conv.std
                x_sample1 = np.expand_dims(x_sample1,axis=0)
                y_hat = model.predict(x_sample1)
                y_prob.append(y_hat)
                y_pred.append(np.argmax(y_hat))
                y_true.append(c)
                x_sample2 = x_sample[9:x_sample.shape[0]]
                x_sample2 = np.concatenate((x_sample2,np.zeros(shape=(18-x_sample.shape[0],20))))
                x = x_sample2
            else:
                x = x_sample
            x = (x-conv.mean)/conv.std
            x = np.expand_dims(x,axis=0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
            
            fn_prob[file] = np.mean(y_prob,axis=0).flatten()
            
    return y_true,y_pred,fn_prob
            
    
    

df = pd.read_csv('C:/Users/98viv/Audio_files.csv')
classes = list(np.unique(df.label))
fn2class = dict(zip(df.file,df.label))
pickle_path = os.path.join('pickle','rnn5_full.p')

with open(pickle_path,'rb') as handle:
    conv = pickle.load(handle)
    
model = load_model(conv.model_path)

y_true,y_pred,fn_prob = build_predictions('C:/Users/98viv/Desktop/speech_hindi/processed')

acc_score = accuracy_score(y_true,y_pred)

y_probs = []
for i,row in df.iterrows():
    y_prob = fn_prob[row.file]
    y_probs.append(y_prob)
    for c,p in zip(classes,y_prob):
        df.at[i,c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('predictions5.csv',index=False)







