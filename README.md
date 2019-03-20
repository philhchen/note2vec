# note2vec
## Organization
utils.py
  - Vocab class, which reads in a data file and compiles information about the notes in there.
  - create_skipgram_dataset(), which creates all the training data that will be fed into the skip-gram model
  - chordsDataset(vocab) creates a dataset of chords given a vocabulary of notes

skipgram.py contains the SkipGram class, which is the wrapper class to take in each "focus note" and its "context notes" and outputs how well the "focus note" predicts the "context notes."

chordrnn.py contains the ChordRNN class, which is a module for running an LSTM language model using chord embeddings

classifier.py contains the ChordClassifier class, which is a module for training and running the classification of chords as major/minor and for their root notes

run.py contains code to feed the training data into the SkipGram, ChordRNN, and ChordClassifier classes, and train the models accordingly
