import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, focus, context):
        embed_focus = self.embeddings(focus)
        embed_focus = embed_focus.view((embed_focus.shape[0], 1, embed_focus.shape[1]))
        embed_context = self.embeddings(context)
        embed_context = embed_context.view((embed_context.shape[0], embed_context.shape[1], 1))
        scores = torch.bmm(embed_focus, embed_context)
        log_probs = F.logsigmoid(scores)
        return log_probs