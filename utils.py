from numpy.random import choice 
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import random

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

	def toInputTensor(self, chords_batch, device=torch.device('cpu')):
		"""
		@param chords_batch (list[list[tuple]]): batch of lists of chords (notes, not indices)
		@param device (torch.device): device to create tensor on
		@returns chords_padded (tensor): padded chords
		@returns list_lengths (list[int]): list of lengths of chorales (descending order)
		"""
		longestChorale = max([len(chorale) for chorale in chords_batch])
		chords_padded = torch.zeros(len(chords_batch), longestChorale, self.largestChord, dtype=torch.long, device=device)
		for i, chorale in enumerate(chords_batch):
			for j, chord in enumerate(chorale):
				for k, note in enumerate(chord):
					chords_padded[i,j,k] = self.w2i[note]
		list_lengths = torch.tensor([len(chorale) for chorale in chords_batch])
		list_lengths, indices = torch.sort(list_lengths, descending=True)
		chords_padded = chords_padded[indices]
		return (chords_padded, list_lengths)

def create_skipgram_dataset(chorales, vocab, batch_size=32, device=torch.device('cpu')):
	"""
	Creates a dataset to train skipgram model
	@param chorales - training data (dim: n_chorales, n_chords)
	@param batch_size - number of chords per training batch
	@param device (torch.device): device to create tensor on
	@returns loader - DataLoader object composed of [chord, target_value]
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
	dataset = TensorDataset(torch.tensor(data, dtype=torch.long, device=device))
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
	return loader

def sortByLength(chords_batch):
	chords_batch.sort(key=lambda x: len(x), reverse=True)

def chordsDataset(vocab):
	chords = []
	note_num = [vocab.i2w[i] for i in range(len(vocab))]
	num_chords = 262

	for i in range(21, 33):
		major = []
		minor = []
		curr = i
		while(curr <= 108):
			major.append(curr)
			curr += 4
			if(curr > 108):
				break
			major.append(curr)
			curr += 3
			if(curr > 108):
				break
			major.append(curr)
			curr += 5
		curr = i
		while(curr <= 108):
			minor.append(curr)
			curr += 3
			if(curr > 108):
				break
			minor.append(curr)
			curr += 4
			if(curr > 108):
				break
			minor.append(curr)
			curr += 5

		for j in range(len(major) - 2):
			if(major[j] in note_num and major[j + 1] in note_num and major[j + 2] in note_num):
				chord = (vocab.w2i[major[j]], vocab.w2i[major[j + 1]], vocab.w2i[major[j + 2]])
				chords.append((chord, 1, vocab.noteNames[i % 12])) # 1 is Major
		for j in range(len(minor) - 2):
			if(minor[j] in note_num and minor[j + 1] in note_num and minor[j + 2] in note_num):
				chord = (vocab.w2i[minor[j]], vocab.w2i[minor[j + 1]], vocab.w2i[minor[j + 2]])
				chords.append((chord, 0, vocab.noteNames[i % 12])) # 0 is minor
	random.shuffle(chords)
	X_train = torch.tensor([chords[i][0] for i in range(num_chords*2 // 3)], dtype=torch.long)
	Y_train = torch.tensor([chords[i][1] for i in range(num_chords*2 // 3)], dtype=torch.float)
	X_test = torch.tensor([chords[i][0] for i in range(num_chords // 3, num_chords)], dtype=torch.long)
	Y_test = torch.tensor([chords[i][1] for i in range(num_chords // 3, num_chords)], dtype=torch.float)
	return X_train, Y_train, X_test, Y_test
