import os

class Downgrade:
    def __init__(self,nfilt=26,nfeat=20,nfft=512,rate=16000):
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('model','rnn5_res1.model')
        self.pickle_path = os.path.join('pickle','rnn5_res1.p')