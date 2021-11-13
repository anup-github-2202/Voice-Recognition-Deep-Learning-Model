from scipy.io import wavfile
import pandas as pd
import numpy as np
from keras.layers import Flatten,Dense,Dropout,LSTM,TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from tqdm import tqdm
from python_speech_features import mfcc
from sklearn.utils.class_weight import compute_class_weight
import warnings
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Downgrade
import os

warnings.filterwarnings("ignore")

def check_data():
    if os.path.isfile(conv.pickle_path):
        print('Exists')
        with open(conv.pickle_path,'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None

def features():
    tmp = check_data()
    if tmp:
        return tmp.data[0],tmp.data[1]
    X = []
    y = []
    minimum = 0.0
    maximum = 0.0
    for i in tqdm(range(samples)):
        choices = np.random.choice(dist_types.index,p=dist_prob)
        file = df.at[np.random.choice(df[df.label == choices].index),'file']
        rate, wav = wavfile.read('C:/Users/98viv/Desktop/speech_hindi/processed/'+choices+'/'+file)
        label = df.at[list(df[df.file == file].index)[0], 'label']
        if(wav.shape[0]>conv.step):
            index = np.random.randint(0,wav.shape[0]-conv.step)
        else:
            continue
        sample = wav[index:index+conv.step]
        X_sample = mfcc(sample,rate,numcep=conv.nfeat,nfilt=conv.nfilt,nfft=conv.nfft)
        minimum = min(np.amin(X_sample),minimum)
        maximum = max(np.amax(X_sample),maximum)
        if(X_sample.shape[0]<9):
            X_sample = np.concatenate((X_sample,np.zeros(shape=(9-X_sample.shape[0],20))))
            X.append(X_sample)
        elif(X_sample.shape[0]>9):
            X_sample1 = X_sample[0:9]
            X.append(X_sample1)
            X_sample2 = X_sample[9:X_sample.shape[0]]
            X_sample2 = np.concatenate((X_sample2,np.zeros(shape=(18-X_sample.shape[0],20))))
            X.append(X_sample2)
            y.append(types.index(label))
        else:
            X.append(X_sample)
        y.append(types.index(label))
    X = np.array(X)
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    conv.mean = mean
    conv.std = std
    print(mean,std)
    X = (X-mean)/std
    y = np.array(y)
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2])
    y = to_categorical(y, num_classes=6)
    conv.data = (X,y)
    with open(conv.pickle_path,'wb') as handle:
        pickle.dump(conv,handle,protocol=2)
    return X, y


def rec_neural():
    model = Sequential()
    model.add(LSTM(128,return_sequences=True,input_shape=shape))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64,activation='relu')))
    model.add(TimeDistributed(Dense(32,activation='relu')))
    model.add(TimeDistributed(Dense(16,activation='relu')))
    model.add(TimeDistributed(Dense(8,activation='relu')))
    model.add(Flatten())
    model.add(Dense(6,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    return model


df = pd.read_csv('C:/Users/98viv/Audio_files.csv')
for i,row in df.iterrows():
    rate, signal = wavfile.read('C:/Users/98viv/Desktop/speech_hindi/processed/'+row['label']+'/'+row['file'])
    df.at[df.index[df['file']==row['file']], 'length'] = signal.shape[0]/rate

types = list(np.unique(df.label))
dist_types = df.groupby(['label'])['length'].mean()
print(dist_types)
samples = 20*int(df['length'].sum())
dist_prob = dist_types/dist_types.sum()
choices = np.random.choice(dist_types.index,p=dist_prob)

conv = Downgrade()
X, y = features()
y_flat = np.argmax(y,axis=1)
shape = (X.shape[1],X.shape[2])
model = rec_neural()
print(X.shape)
class_weight = compute_class_weight('balanced',np.unique(y_flat),y_flat)

checkpoint = ModelCheckpoint(conv.model_path,monitor='val_acc',verbose=1,mode='max',
                             save_best_only=True,save_weights_only=False,period=1)

model.fit(X,y,epochs=10,batch_size=32,shuffle=True,validation_split=0.2,callbacks=[checkpoint])

model.save(conv.model_path)