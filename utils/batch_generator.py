import pickle
import numpy as np
from pathlib import Path


class PathContextReader:
    ''' class for preprocessing the data '''
    def __init__(self, path):
        self.bags_train = None
        self.bags_test  = None

        self.path = Path(path).resolve()        

    def read_path_contexts(self):
        self.read_dictionaries()

        self.bags_train = self.read_data(data_path="train.txt")
        self.bags_test  = self.read_data(data_path="test.txt")

        print("Number of unique of words: " + str(len(self.word_count)))
        print("Number of unique of paths: " + str(len(self.path_count)))
        print("Number of unique of targets: " + str(len(self.target_count)))

        print("Number of training samples: " + str(len(self.bags_train)))
        print("Number of testing samples: " + str(len(self.bags_test)))

    def read_dictionaries(self):
        with open(str(self.path / 'reduced_word_count.pkl'), 'rb') as f:
            self.word_count = pickle.load(f)
        with open(str(self.path / 'reduced_word2idx.pkl'), 'rb') as f:
            self.word2idx = pickle.load(f)
        with open(str(self.path / 'reduced_idx2word.pkl'), 'rb') as f:
            self.idx2word = pickle.load(f)

        with open(str(self.path / 'reduced_path_count.pkl'), 'rb') as f:
            self.path_count = pickle.load(f)
        with open(str(self.path / 'reduced_path2idx.pkl'), 'rb') as f:
            self.path2idx = pickle.load(f)
        with open(str(self.path / 'reduced_idx2path.pkl'), 'rb') as f:
            self.idx2path = pickle.load(f)

        with open(str(self.path / 'reduced_target_count.pkl'), 'rb') as f:
            self.target_count = pickle.load(f)
        with open(str(self.path / 'reduced_target2idx.pkl'), 'rb') as f:
            self.target2idx = pickle.load(f)
        with open(str(self.path / 'reduced_idx2target.pkl'), 'rb') as f:
            self.idx2target = pickle.load(f)

    def read_data(self, data_path="train.txt"):
        bags=[]

        with open((self.path / data_path), 'r') as file:
            for function_line in file:
                splited_function_line = function_line.split(" ")
                label_ids = splited_function_line[0]
                triples = splited_function_line[1:]
                triple_ids = []

                for triple in triples:
                    splited_triple = triple.split('\t')
                    if len(splited_triple) != 3: 
                        assert False, "Weird non-triple data row."

                    e1, p, e2 = int(splited_triple[0]), int(splited_triple[1]), int(splited_triple[2])
                    triple_ids.append([e1,p,e2])

                bags.append((label_ids, triple_ids))

        return bags
        
class Generator:

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        
        self.number_of_batch = len(data) // self.batch_size
        self.random_ids = np.random.permutation(len(data))
        
        self.batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        data = self.data
        pos_start = self.batch_size * self.batch_idx
        pos_end   = self.batch_size * (self.batch_idx+1)

        raw_data = np.asarray([data[x][1] for x in self.random_ids[pos_start:pos_end]])
        raw_tags = np.asarray([int(data[x][0]) for x in self.random_ids[pos_start:pos_end]])
        
        self.batch_idx += 1
        if  self.batch_idx == self.number_of_batch:
            self.batch_idx = 0

        return raw_data, raw_tags