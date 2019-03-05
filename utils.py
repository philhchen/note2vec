from numpy.random import choice 
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

class Vocab():
	def __init__(self, data=None):
		self.vocab = set([0]) # set of all notes processed
		self.counts = defaultdict(int) # number of counts for each note
		self.totalCounts = 0 # total number of note processed
		self.probs = [] # probability of each note in vocab
		self.w2i = {0:0} # note to index
		self.i2w = {0:0} # index to note
		self.noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
		self.numChords = 0
		self.largestChord = 0
		self.numPairs = 0
		if data != None:
			self.updateVocab(data)

	def __len__(self):
		return len(self.vocab)

	def updateVocab(self, data):
		"""
		Updates the class given new data
		@param data (dim: n_datasets, n_chorales, n_chords, n_notes)
		"""
		for dataset in data:
			for chorale in data[dataset]:
				for chord in chorale:
					self.numChords += 1
					self.largestChord = max(self.largestChord, len(chord))
					self.numPairs += len(chord) * (len(chord) - 1)
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
	"""
	Creates a dataset to train skipgram model
	@param chorales - training data (dim: n_chorales, n_chords)
	@param batch_size - number of chords per training batch
	returns loader - DataLoader object composed of [chord, target_value]
	"""
	data = np.zeros([vocab.numChords * 2, vocab.largestChord + 1])
	count = 0
	for ch in chorales:
		for c in ch:
			data[count, -1] = 1
			data[count + 1, -1] = 0
			for i, n in enumerate(c):
				data[count, i] = vocab.w2i[n]
				data[count + 1, i] = vocab.getNegativeSample()
			count += 2
	dataset = TensorDataset(torch.tensor(data, dtype=torch.long))
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
	return loader

