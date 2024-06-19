import os

current_dir = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(current_dir, 'vocab.txt')

class Vocab(object):

    def __init__(self, vocab_path):
        vocab_list = []
        for line in open(vocab_path):
            p, _, _ = line.partition(':')
            vocab_list.append(p)
        self.vocab = vocab_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        #self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        if smiles in self.vmap.keys():
            return self.vmap[smiles]
        else:
            return 0

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)
    
vocab = Vocab(vocab_path)