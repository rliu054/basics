import torch
import os

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
    

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        
    def get_data(self, path, batch_size=20):
        # add words to dict
        with open(path, 'r') as f:
            num_tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                num_tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        
        # tokenize the file content
        ids = torch.LongTensor(num_tokens)
        token_idx = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token_idx] = self.dictionary.word2idx[word]
                    token_idx += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size] # discarding
        return ids.view(batch_size, -1)