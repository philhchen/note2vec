from skipgram import SkipGram
from utils import Vocab, create_skipgram_dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch
import numpy as np
import pickle

# Hyperparameters
embed_size = None
batch_size = 32
learning_rate = 0.01
n_epoch = 50
average = True
simple = False

# File handling
load_prev = False
save = True
data_file = 'data/jsb-chorales-quarter.pkl'
model_file = 'results/model{}_simple.bin'.format(embed_size)
embeddings_file = 'results/embeddings{}_simple.tsv'.format(embed_size)
meta_file = 'results/meta.tsv'
loss_file = 'results/model{}_simple.loss'.format(embed_size)

def train_skipgram(vocab, sg_loader):
	losses = []
	loss_fn = nn.L1Loss()
	model = SkipGram(len(vocab), embed_size, simple)
	print(model)

	if load_prev:
		try:
			model.load_state_dict(torch.load(model_file))
		except:
			print('Could not load file')

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
		if save:
			save_params(vocab, model, losses)
	return model, losses

def save_params(vocab, model, losses):
	torch.save(model.state_dict(), model_file)
	embeddings = np.array(model.embeddings.weight.data) if model.simple else np.array(model.embedding_mat.data * model.mask)
	with open(embeddings_file, 'w') as f:
		for i in range(len(embeddings)):
			for j in range(len(embeddings[0])):
				f.write('{}\t'.format(embeddings[i,j]))
			f.write('\n')

	with open(meta_file, 'w') as f:
		for i in range(len(vocab)):
			f.write('{}\n'.format(vocab.num2note(vocab.i2w[i])))

	with open(loss_file, 'w') as f:
		for loss in losses:
			f.write('{}\n'.format(loss))

def main():
	with open(data_file, 'rb') as f:
		data = pickle.load(f)
	vocab = Vocab(data)
	sg_loader = create_skipgram_dataset(chorales=data['train'], vocab=vocab, batch_size=batch_size)
	sg_model, sg_losses = train_skipgram(vocab, sg_loader)

if __name__ == '__main__':
	main()
