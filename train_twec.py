import os
from twec.twec import TWEC
from gensim.models.word2vec import Word2Vec
import time


aligner = TWEC(size=300, siter=10, diter=10, workers=4)

start = time.time()
# train the compass: the text should be the concatenation of the text from the slices
aligner.train_compass("./data/compass.txt", overwrite=True) # keep an eye on the overwrite behaviour
end = time.time()
print("Time Taken for TWEC Pre-Training:",(end-start)," ms")

slices = {}
for file in os.listdir('./data/'):
    if file !='compass.txt':
        start = time.time()
        slices[file.split('.')[0]] = aligner.train_slice(f'./data/{file}')
        end = time.time()
        print("Time Taken for TWEC Fine-tuning:",(end-start)," ms")