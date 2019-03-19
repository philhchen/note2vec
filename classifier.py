import torch
import torch.nn as nn

class ChordClassifier(nn.Module):
	def __init__(self, embeddings, n_channels=32, k=[13, 10, 7], file=None):
		"""
		@param embeddings (SkipGram object)
		"""
		super(ChordClassifier, self).__init__()
		self.embed_size = embeddings.embed_size
		self.embeddings = embeddings

		self.conv = []
		self.pool = []
		for kernel in k:
			self.conv.append(nn.Conv1d(1, n_channels, kernel))
			self.pool.append(nn.MaxPool1d(self.embed_size - kernel + 1))
		self.dropout = nn.Dropout(p=0.0)

		self.proj = nn.Linear(n_channels * len(k), 1)
		self.sig = nn.Sigmoid()
		self.loss = nn.BCELoss()

		if file != None:
			try:
				self.load_state_dict(torch.load(file))
			except:
				print('Could not load file')

	def forward(self, chord, ans, use_emb=True):
		"""
		@param chord Tensor(ChordLength)
		@param ans (Float) 1 or 0 - 1 if major, 0 if minor
		@param use_emb (Bool): If set to True, uses chord embeddings; else baseline model
		@returns pred: probability that the chord is major
		@returns loss: binary cross-entropy loss
		@returns correct: bool whether predicted value was correct
		"""
		if use_emb:
			chord_emb = self.embeddings.chordEmbedding(chord).view(1, 1, -1)
		else:
			chord_emb = torch.zeros(1, 1, self.embed_size)
			for i in chord:
				chord_emb[0,0,i] = 1

		conv_out = []
		for i in range(len(self.conv)):
			curr = self.conv[i](chord_emb)
			curr = self.pool[i](curr).view(1, 1, -1)
			conv_out.append(curr)
		conv_out = torch.cat(conv_out, 2)
		conv_out = self.dropout(conv_out)

		pred = self.sig(self.proj(conv_out)).squeeze()
		loss = self.loss(pred, ans)
		correct = abs(pred - ans) < 0.5
		return pred, loss, correct