from skipgram import SkipGram
from utils import Vocab, create_skipgram_dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch
import numpy as np
import pickle

# Hyperparameters
embed_size = 8
batch_size = 32
learning_rate = 0.01
n_epoch = 20

# File handling
load_prev = True
data_file = 'data/jsb-chorales-quarter.pkl'
model_file = 'results/model{}.bin'.format(embed_size)
embeddings_file = 'results/embeddings{}.tsv'.format(embed_size)
meta_file = 'results/meta.tsv'

def train_skipgram(vocab, sg_loader):
	losses = []
	loss_fn = nn.L1Loss()
	model = SkipGram(len(vocab), embed_size)
	print(model)

	if load_prev:
		model.load_state_dict(torch.load(model_file))

	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	for epoch in range(n_epoch):
		total_loss = 0.0
		for i, sample_batched in enumerate(sg_loader):
			sample_batched = sample_batched[0]
			in_w_var = Variable(sample_batched[:,0])
			ctx_w_var = Variable(sample_batched[:,1])
			# print(in_w_var.shape)
			model.zero_grad()
			log_probs = model(in_w_var, ctx_w_var)
			loss = loss_fn(log_probs, Variable(sample_batched[:,2].float()))

			loss.backward()
			optimizer.step()

			total_loss += loss.data

		losses.append(total_loss.item())
		print('Epoch:', epoch, 'Loss:', total_loss.item())
		save_model(vocab, model)
	return model, losses

def save_params(vocab, model):
	torch.save(model.state_dict(), model_file)
	embeddings = np.array(model.embeddings.weight.data)
	with open(embeddings_file, 'w') as f:
		for i in range(len(embeddings)):
			f.write('{}\t{}\t{}\t{}\n'.format(embeddings[i,0], embeddings[i,1], embeddings[i,2], embeddings[i,3]))

	with open(meta_file, 'w') as f:
		for i in range(len(i2w)):
			f.write('{}\n'.format(num2note(i2w[i])))

def main():
	with open(data_file, 'rb') as f:
		data = pickle.load(f)
	vocab = Vocab(data)
	sg_loader = create_skipgram_dataset(chorales=data['train'], vocab=vocab, batch_size=32)
	sg_model, sg_losses = train_skipgram(vocab, sg_loader)

if __name__ == '__main__':
	main()
