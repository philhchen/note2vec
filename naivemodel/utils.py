from numpy.random import choice 
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

class Vocab():
	def __init__(self, data=None):
		self.vocab = set() # set of all notes processed
		self.counts = defaultdict(int) # number of counts for each note
		self.totalCounts = 0 # total number of note processed
		self.probs = [] # probability of each note in vocab
		self.w2i = {} # note to index
		self.i2w = {} # index to note
		self.noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
		if data != None:
			self.updateVocab(data)

	def __len__(self):
		return len(self.vocab)

	def updateVocab(self, data):
		for dataset in data:
			for chorale in data[dataset]:
				for chord in chorale:
					for note in chord:
						self.vocab.add(note)
						if note not in self.w2i:
							self.w2i[note] = len(self.w2i)
							self.i2w[len(self.i2w)] = note
						self.counts[self.w2i[note]] += 1
		self.vocab_size = len(self.vocab)
		self.totalCounts = sum([self.counts[k] for k in self.counts])
		self.probs = [self.counts[k]/self.totalCounts for k in self.counts]

	def num2note(self, num):
		note_name = self.noteNames[num % 12]
		octave = str((num-12)//12)
		return note_name + octave

	def getNegativeSample(self):
		return choice(list(self.counts.keys()), p=self.probs)

def create_skipgram_dataset(chorales, vocab, batch_size=32):
	data = []
	for ch in chorales:
		for c in ch:
			for n in c:
				data += [(vocab.w2i[n], vocab.w2i[n1], 1) for n1 in c if n1 != n]
				# negative sample
				data += [(vocab.w2i[n], vocab.getNegativeSample(), 0) for _ in range(3)]

	dataset = TensorDataset(torch.tensor(data, dtype=torch.long))
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
	return loader

