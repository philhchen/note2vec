
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class ChordRNN(nn.Module):
	def __init__(self, vocab, embeddings, cfg, file=None):
		"""
		@param vocab (Vocab): Vocab object containing information about training data notes
		@param embeddings (SkipGram): SkipGram object containing note2vec and chord2vec embeddings
		@param cfg (Config): Config object containing hyperparameters
		"""
		super(ChordRNN, self).__init__()
		self.vocab = vocab
		self.embeddings = embeddings
		self.lstm = nn.LSTM(input_size=embeddings.embed_size, hidden_size=cfg.nHidden, num_layers=cfg.nLayers, dropout=cfg.dropout, bidirectional=True)
		self.proj = nn.Linear(2*cfg.nHidden, embeddings.embed_size, bias=False)
		self.loss = nn.MSELoss()
		self.device = cfg.device
		self.load_state_dict(torch.load(file))
		if file != None:
			try:
				self.load_state_dict(torch.load(file))
			except:
				print('Could not load file')

	def forward(self, source):
		"""
		Applies lstm model to sequence
		@param source (List[List[Tuple]]): batch of lists of chords of size nbatches x nchords
		"""
		src_padded, list_lengths = self.vocab.toInputTensor(source, self.device)
		list_lengths = [i-1 for i in list_lengths]
		total_embed = self.embeddings.chordEmbedding(src_padded).permute(1,0,2)
		src_embed = total_embed[:-1]
		tgt_embed = total_embed[1 : list_lengths[0]+1]

		src_embed = pack_padded_sequence(src_embed, list_lengths)
		hiddens, _ = self.lstm(src_embed)
		hiddens, _ = pad_packed_sequence(hiddens)
		out_proj = F.relu(self.proj(hiddens))
		scores = self.loss(out_proj, tgt_embed)

		return (scores, out_proj, tgt_embed)

	def decodeGreedy(self, primer, numSteps):
		"""
		Greedy decoding for generating music
		@param primer (List[Tuple])
		@param numSteps (int)
		"""
		curr, _ = self.vocab.toInputTensor([primer])
		curr = self.embeddings.chordEmbedding(curr).permute(1,0,2)
		output_chords = []
		for i in range(numSteps):
			hiddens, states = self.lstm(curr, states) if i > 0 else self.lstm(curr)
			curr = F.relu(self.proj(hiddens[-1:]))
			output_chords.append(curr.squeeze())
		return output_chords

class Config:
	def __init__(self, nLayers=3, nHidden=64, dropout=0.2, device=torch.device('cpu')):
		self.nLayers = nLayers
		self.nHidden = nHidden
		self.dropout = dropout
		self.device = device

