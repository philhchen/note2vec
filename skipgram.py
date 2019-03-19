import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size=None, simple=False, file=None):
        """
        Initializes SkipGram object
        @param vocab_size - size of the input vocabulary to be embedded
        @param embed_size - (optional) dimension of the note/chord embeddings
        @param simple - (optional) Uses overtone-based embeddings if true, regular embeddings if false
        """
        super(SkipGram, self).__init__()
        if embed_size == None:
            embed_size = vocab_size + 36
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.simple = simple
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embedding_mat = nn.Parameter(torch.ones(vocab_size, embed_size))
        self.mask = self.createMask(simple)
        
        if file != None:
            try:
                self.load_state_dict(torch.load(file))
            except:
                print('Could not load file')
    
    def forward(self, chords, average=True):
        """
        Forward pass of training
        @param chords - the input chords to use for training (dim: batch_size x chord_size)
        @returns log_prob - float equal to the sum of log likelihoods of the input chords
        """
        contexts = torch.zeros(chords.shape[0] * chords.shape[1], chords.shape[1] - 1, dtype=torch.long)
        count = 0
        for chord in chords:
            for focus in chord:
                ctx = [context for context in chord if context != focus]
                contexts[count,:len(ctx)] = torch.tensor(ctx)
                count += 1

        if self.simple:
            embed_focus = self.embeddings(chords)
            embed_contexts = self.embeddings(contexts)
        else:
            embed_focus = F.embedding(chords, self.embedding_mat*self.mask)
            embed_contexts = F.embedding(contexts, self.embedding_mat*self.mask)

        embed_focus = embed_focus.view((-1, 1, embed_focus.shape[2]))
        embed_contexts = torch.mean(input=embed_contexts, dim=1)
        embed_contexts = embed_contexts.view((embed_contexts.shape[0], embed_contexts.shape[1], 1))
        
        scores = torch.bmm(embed_focus, embed_contexts)
        log_prob = F.logsigmoid(scores)
        return log_prob

    def chordEmbedding(self, chords, average=True):
        """
        Forward pass of chord embedding
        @param chords - the input chords to embed (dim: numChords, numNotesPerChord)
        @returns embed_chords, the embeddings of the input chords (dim: input_dim, embed_size)
        """
        if self.simple:
            embed_notes = self.embeddings(chords)
        else:
            embed_notes = F.embedding(chords, self.embedding_mat*self.mask)
        embed_chords = torch.mean(input=embed_notes, dim=-2)
        return embed_chords

    def createMask(self, simple):
        """
        Creates mask over the embeddings for when simple = False
        @param simple - False if real masks are desired
        @param vocab_size - Size of the input vocab
        @param embed_size - Dimension of vector embeddings
        @returns mask (dim: vocab_size, embed_size)
        """
        if simple:
            return torch.ones([self.vocab_size, self.embed_size])

        mask = torch.zeros([self.vocab_size, self.embed_size], dtype=torch.float)
        for i in range(1, self.vocab_size):
            for n, j in enumerate([0, 12, 19, 24, 28, 31, 34, 36]):
                if i + j < self.embed_size:
                    mask[i, i+j] = 1
        return mask

    def vec2chord(self, chord_embedding, vocab):
        """
        Greedily reverses embedding of chord
        @param embed - Tensor of embedding of a chord (dim: embed_size)
        @param vocab - Vocab object containing i2w dict for decoding indices to notes
        @returns chord - tuple of 
        """
        notesPerChord = 4
        chord = []
        emb = (self.embedding_mat*self.mask).unsqueeze(-1)
        chord_embedding = chord_embedding.repeat(self.vocab_size, 1, 1)

        for i in range(notesPerChord):
            scores = torch.bmm(chord_embedding, emb).squeeze()
            # print(scores)
            _, inds = scores.sort(descending=True)
            for ind in inds:
                note = vocab.i2w[ind.item()]
                if note not in chord:
                    chord.append(note)
                    break
            projection = (scores[ind] * emb[ind] / (torch.norm(emb[ind]) * torch.norm(emb[ind]))).squeeze()
            chord_embedding -= projection
        return tuple(chord)

