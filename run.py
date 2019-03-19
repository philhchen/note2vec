from skipgram import SkipGram
from utils import Vocab, create_skipgram_dataset, chordsDataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch
import numpy as np
import pickle
from chordrnn import ChordRNN, Config
from classifier import ChordClassifier

# Configuration
trainSkipgram = True
trainRNN = False
trainClassifier = False

# Hyperparameters
embed_size = None
batch_size = 32
learning_rate = 0.01
n_epoch = 200
average = True
simple = False
use_emb = True

# File handling
load_emb = True
load_rnn = False
load_cnn = False
save_emb = True
save_rnn = False
save_cnn = False
data_file = 'data/jsb-chorales-quarter.pkl'

rnn_bin = 'results/rnn/model{}_avg.bin'.format(embed_size)
rnn_loss = 'results/rnn/model{}_avg.loss'.format(embed_size)

cnn_bin = 'results/classifier/model{}_avg.bin'.format(embed_size)
cnn_loss = 'results/classifier/model{}_avg.loss'.format(embed_size)

embeddings_bin = 'results/embeddings/model{}_avg.bin'.format(embed_size)
embeddings_tsv = 'results/embeddings/embeddings{}_avg.tsv'.format(embed_size)
embeddings_meta = 'results/embeddings/meta.tsv'
embeddings_meta2 = 'results/embeddings/meta2.tsv'
embeddings_loss = 'results/embeddings/model{}_avg.loss'.format(embed_size)

def train_skipgram(vocab, sg_loader):
	losses = []
	loss_fn = nn.L1Loss()
	model = SkipGram(len(vocab), file=embeddings_bin) if load_emb else SkipGram(len(vocab), embed_size, simple)
	print(model)

	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	for epoch in range(n_epoch):
		total_loss = 0.0
		for i, sample_batched in enumerate(sg_loader):
			sample_batched = sample_batched[0]

			model.zero_grad()
			log_probs = model(sample_batched[:,:-1], average)
			loss = loss_fn(log_probs, Variable(sample_batched[:,-1].float()))

			loss.backward()
			optimizer.step()

			total_loss += loss.data

		losses.append(total_loss.item())
		print('Epoch:', epoch, 'Loss:', total_loss.item())
		save_params(emb=model, losses=losses, vocab=vocab)
		# Early stopping
		if len(losses) > 2 and losses[-1] > losses[-2]:
			break
	return model, losses

def train_chordRNN(vocab, data):
	cfg = Config()
	model = SkipGram(len(vocab), file=embeddings_bin) if load_emb else SkipGram(len(vocab), embed_size, simple)
	rnn = ChordRNN(vocab, model, cfg, file=rnn_bin) if load_rnn else ChordRNN(vocab, model, cfg)

	losses = []
	optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

	for epoch in range(n_epoch):
		total_loss = 0.0
		for i in range(len(data)//batch_size):
			loss, _, _ = rnn(data[i*batch_size : (i+1)*batch_size])
			loss.backward()
			optimizer.step()
			total_loss += loss

		print('Epoch:', epoch, 'Loss:', total_loss.item())
		losses.append(total_loss.item())
		save_params(rnn=rnn, losses=losses)
		# Early stopping
		if len(losses) > 2 and losses[-1] > losses[-2]:
			break
	
	out = rnn.decodeGreedy(data[0][0:2], 1)
	print([model.vec2chord(o, vocab) for o in out])

def train_classifier(vocab):
	embeddings = SkipGram(len(vocab), file=embeddings_bin) if load_emb else SkipGram(len(vocab), embed_size, simple)
	model = ChordClassifier(embeddings, file=cnn_bin) if load_cnn else ChordClassifier(embeddings)

	losses = []
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	X_train, Y_train, X_test, Y_test = chordsDataset(vocab)

	prev_acc = 0
	acc = 0

	for epoch in range(n_epoch):
		total_loss = 0.0
		for i, x in enumerate(X_train):
			_, loss, _ = model(x, Y_train[i], use_emb=use_emb)
			loss.backward()
			optimizer.step()
			total_loss += loss
		total_loss /= len(X_train)
		print('Epoch:', epoch, 'Loss:', total_loss.item())
		losses.append(total_loss.item())

		if epoch % 10 == 9:
			acc = 0
			for i, x in enumerate(X_test):
				_, _, correct = model(x, Y_test[i])
				acc += correct.type(torch.float)
			acc /= len(X_test)
			print('Test accuracy:', acc.item())

			prev_acc = acc
			save_params(cnn=model, losses=losses)

def save_params(emb=None, rnn=None, cnn=None, losses=None, vocab=None):
	if save_emb and emb != None:
		torch.save(emb.state_dict(), embeddings_bin)
		embeddings = np.array(emb.embeddings.weight.data) if emb.simple else np.array(emb.embedding_mat.data * emb.mask)

		with open(embeddings_tsv, 'w') as f:
			for i in range(len(embeddings)):
				for j in range(len(embeddings[0])):
					f.write('{}\t'.format(embeddings[i,j]))
				f.write('\n')

		if vocab != None:
			with open(embeddings_meta, 'w') as f:
				f.write('{}\n'.format(0))
				for i in range(vocab.vocablist[1], vocab.vocablist[-1]+1):
					f.write('{}\n'.format(vocab.num2note(i)))

		if vocab != None:
			with open(embeddings_meta2, 'w') as f:
				for i in vocab.i2w:
					f.write('{}\n'.format(vocab.i2w[i]))

		if losses != None:
			with open(embeddings_loss, 'w') as f:
				for loss in losses:
					f.write('{}\n'.format(loss))

	if save_rnn and rnn != None:
		torch.save(rnn.state_dict(), rnn_bin)

		if losses != None:
			with open(rnn_loss, 'w') as f:
				for loss in losses:
					f.write('{}\n'.format(loss))

	if save_cnn and cnn != None:
		torch.save(cnn.state_dict(), cnn_bin)

		if losses != None:
			with open(cnn_loss, 'w') as f:
				for loss in losses:
					f.write('{}\n'.format(loss))
		

def main():
	with open(data_file, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		data = u.load()
	vocab = Vocab(data)

	if trainSkipgram:
		sg_loader = create_skipgram_dataset(chorales=data['train'], vocab=vocab, batch_size=batch_size)
		sg_model, sg_losses = train_skipgram(vocab, sg_loader)
	if trainRNN:
		train_chordRNN(vocab, data['train'])
	if trainClassifier:
		train_classifier(vocab)

if __name__ == '__main__':
	main()
