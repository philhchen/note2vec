import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size=None, simple=False):
        super(SkipGram, self).__init__()
        if embed_size == None:
            embed_size = vocab_size + 36
        self.simple = simple
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.embedding_mat = nn.Parameter(torch.rand(vocab_size, embed_size))
        self.mask = self.createMask(simple, vocab_size, embed_size)
    
    def forward(self, focus, context):
        if self.simple:
            embed_focus = self.embeddings(focus)
            embed_context = self.embeddings(context)
        else:
            embed_focus = F.embedding(focus, self.embedding_mat*self.mask)
            embed_context = F.embedding(context, self.embedding_mat*self.mask)

        embed_focus = embed_focus.view((embed_focus.shape[0], 1, embed_focus.shape[1]))
        embed_context = embed_context.view((embed_context.shape[0], embed_context.shape[1], 1))
        scores = torch.bmm(embed_focus, embed_context)
        log_probs = F.logsigmoid(scores)
        return log_probs

    def createMask(self, simple, vocab_size, embed_size):
        if simple:
            return torch.ones([vocab_size, embed_size])

        mat = torch.zeros([vocab_size, embed_size], dtype=torch.float)
        for i in range(vocab_size):
            for n, j in enumerate([0, 12, 19, 24, 28, 31, 34, 36]):
                if i + j < embed_size:
                    mat[i, i+j] = 1
        return mat