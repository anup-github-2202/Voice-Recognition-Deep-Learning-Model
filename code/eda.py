import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc
import librosa

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2,ncols=3,sharey=True,figsize=(20,5))
    fig.suptitle('Time Series',size=16)
    i = 0
    for x in range(2):
        for y in range(3):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1
            
def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2,ncols=3,sharey=True,figsize=(20,5))
    fig.suptitle('Fourier Transforms',size=16)
    i = 0
    for x in range(2):
        for y in range(3):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1
           
def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2,ncols=3,sharey=True,figsize=(20,5))
    fig.suptitle('MFCCs',size=16)
    i = 0
    for x in range(2):
        for y in range(3):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],cmap='hot',interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i+=1
            
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

def calc_fft(signal, rate):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/rate)
    magnitude = abs(np.fft.rfft(signal)/n)
    return (magnitude,freq)


df = pd.read_csv('C:/Users/98viv/Audio_files.csv')
for i,row in df.iterrows():
    rate, signal = wavfile.read('C:/Users/98viv/Desktop/speech_hindi/'+row['label']+'/'+row['file'])
    df.at[df.index[df['file']==row['file']], 'length'] = signal.shape[0]/rate
  
types = list(np.unique(df.label))
dist_types = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Distribution', y = 1.1)
ax.pie(dist_types,labels=dist_types.index,autopct='%1.1f%%',shadow=False,startangle=90)
ax.axis('equal')
plt.show()

signals = {}
fft = {}
fbank = {}
mfccs = {}

for t in types:
    wav_file = df[df.label == t].iloc[0,0]
    signal, rate = librosa.load('C:/Users/98viv/Desktop/speech_hindi/'+t+'/'+wav_file,sr=48000)
    mask = envelope(signal, rate, 0.005)
    signal = signal[mask]
    signals[t] = signal
    fft[t] = calc_fft(signal,rate)
    mfccs[t] = mfcc(signal[:rate],rate,numcep=20,nfilt=26,nfft=1200).T
    
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_mfccs(mfccs)
plt.show()


for i,row in df.iterrows():
    signal, rate = librosa.load('C:/Users/98viv/Desktop/speech_hindi/'+row['label']+'/'+row['file'],
                                    sr=16000)
    print(rate)
    mask = envelope(signal,rate,0.005)
    wavfile.write(filename='C:/Users/98viv/Desktop/speech_hindi/processed/'+row['label']+'/'+row['file'],
                  rate=rate,data=signal[mask])
    

    
    
    
    
    
    
    
    
    
    
    