import os
import csv


path = 'C:/Users/98viv/Desktop/speech_hindi'
dic = {'aage':1,'piche':2,'dayen':3,'bayen':4,'chalo':5,'ruko':6}
#os.remove("C:/Users/98viv/Audio_files.csv")
with open('Audio_files.csv', 'w',newline = "") as f:
  writer = csv.writer(f)
  writer.writerow(['file','label'])
  folders = ['aage','bayen','chalo','dayen','piche','ruko']
  for folder in folders:
      filenames = os.listdir(os.path.join(path,folder))
      for filename in filenames:
          writer.writerow([filename,folder])
'''        
          
path = 'C:/Users/98viv/Desktop/testing'
with open('Audio_files_test.csv', 'w',newline = "") as f:
  writer = csv.writer(f)
  writer.writerow(['file'])
  for filename in os.listdir(path):
      writer.writerow([filename])
          '''