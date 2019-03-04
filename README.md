# note2vec
## Organization
utils.py contains the Vocab class, which reads in a data file and compiles information about the notes in there.

utils.py also contains create_skipgram_dataset(), which creates all the training data that will be fed into the skip-gram model. It might be useful to play with the output of create_skipgram_dataset to get a sense of what it does.

skipgram.py contains the SkipGram class, which is the wrapper class to take in each "focus note" and its "context notes" and outputs how well the "focus note" predicts the "context notes."

run.py contains code to feed the training data into the SkipGram class and trains the model accordingly

## Next steps
If you plug in the result files (embeddings8.tsv and meta.tsv) into tensorflow projector, the results aren't too great and the model doesn't seem to learn any music theory. We should try two different training methods and see if they work better:

1. For each chord (a,b,c,d), we feed in {(a,b,1), (a,c,1), (a,d,1), (b,a,1),...} as training data. Maybe we should try feeding in {(a, (b+c+d)/3, 1), (b, (a+c+d)/3, 1), (c, (a+b+d)/3, 1), (d, (a+b+c)/3, 1)} as training data. The intuition is to match each note with its context "chord" all together.

2. Do both note2vec and chord2vec simultaneously
