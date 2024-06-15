import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import contractions
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm


def read_csv(file_path):
    df = pd.read_csv(file_path)
    text = df['Description'].tolist()
    labels = df['Class Index'].tolist()
    return text, labels

def tokenize(text):
    text = contractions.fix(text)
    sentences = sent_tokenize(text)
    new_sents = []
    for sentence in sentences:
        sentence = re.sub(r'[^\w\s]', '', sentence)
        sentence = re.sub(r'\b\d+\b', 'num', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        new_sents.append(sentence)
    tokenized_text = [['<s>']+word_tokenize(t)+['</s>'] for t in new_sents] 
    return tokenized_text
    


train_text, train_labels = read_csv('ANLP-2/train.csv')
test_text, test_labels = read_csv('ANLP-2/test.csv')
print("Tokenizing....")
sents = []
for sample in train_text:
    sents += tokenize(sample)
print("Done")



START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

class LSTMDataset(Dataset):
  def __init__(self, data: list[tuple[list[str], list[int]]], vocabulary_words:Vocab|None=None, vocabulary_tags:Vocab|None=None):
    """Initialize the dataset. Setup Code goes here"""
    self.sentences = data # list of sentences

    if vocabulary_words is None:
      self.vocabulary_words = build_vocab_from_iterator(self.sentences, min_freq=3, specials=[PAD_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN]) # use min_freq for handling unkown words better
      self.vocabulary_words.set_default_index(self.vocabulary_words[UNKNOWN_TOKEN])
    else:
      self.vocabulary_words = vocabulary_words

  def __len__(self) -> int:
    """Returns number of datapoints."""
    return len(self.sentences)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the datapoint at `index`."""
    sentence_in = self.sentences[index][:-1]
    sentence_out = self.sentences[index][1:]
    return torch.tensor(self.vocabulary_words.lookup_indices(sentence_in)), torch.tensor(self.vocabulary_words.lookup_indices(sentence_out)) 

  def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a list of datapoints, batch them together"""
    sentences_in = [i[0] for i in batch]
    sentences_out = [i[1] for i in batch]
    padded_sentences_in = pad_sequence(sentences_in, batch_first=True, padding_value=self.vocabulary_words[PAD_TOKEN]) # pad sentences with pad token id
    padded_sentences_out = pad_sequence(sentences_out, batch_first=True, padding_value=self.vocabulary_words[PAD_TOKEN]) # pad sentences with pad token id

    return padded_sentences_in, padded_sentences_out

dataset = LSTMDataset(sents)
train_size, val_size = 0.8, 0.2
batch_size = 64

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)


class Elmo(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim=256, return_embeddings=False):
        super(Elmo, self).__init__()
        self.return_embeddings = return_embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, hidden_size)
        self.forward_lstm_1 = torch.nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.forward_lstm_2 = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.forward_fc =  torch.nn.Linear(hidden_size, vocab_size)

        self.backward_lstm_1 = torch.nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.backward_lstm_2 = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.backward_fc =  torch.nn.Linear(hidden_size, vocab_size)
        
        
    def forward(self, seq1, seq2):
        # Forward LM
        forward_emd = self.embedding(seq1)
        forward_lstm_1_h0, _ = self.forward_lstm_1(forward_emd)
        forward_lstm_2_h0, _ = self.forward_lstm_2(forward_lstm_1_h0)
        forward_fc_out = self.forward_fc(forward_lstm_2_h0)

        # Backward LM
        backward_emd = self.embedding(seq2)
        reversed_embedded = torch.flip(backward_emd, [1])
        backward_lstm_1_h0, _ = self.backward_lstm_1(reversed_embedded)
        backward_lstm_2_h0, _ = self.backward_lstm_2(backward_lstm_1_h0)
        backward_lstm_2_h0 = torch.flip(backward_lstm_2_h0, [1])
        backward_fc_out = self.backward_fc(backward_lstm_2_h0)
        if self.return_embeddings:
            embeddings_concat = torch.cat((forward_emd, backward_emd), dim=-1)
            lstm_layer1 = torch.cat((forward_lstm_1_h0, torch.flip(backward_lstm_1_h0, [1])), dim=-1)
            lstm_layer2 = torch.cat((forward_lstm_2_h0, backward_lstm_2_h0), dim=-1)
            print(forward_emd.shape)
            print(embeddings_concat.shape, lstm_layer1.shape, lstm_layer2.shape)
        else:
            return forward_fc_out, backward_fc_out


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), self.path)


model = Elmo(len(dataset.vocabulary_words), 256)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Utilising",device)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.vocabulary_words[PAD_TOKEN]) # use ignore index to ignore losses for padding value indices
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model = model.to(device)
model_path = 'models/Elmo.pt'
early_stopping = EarlyStopping(patience=5, path=model_path)

for epoch_num in range(25):
  model.train()
  epoch_loss = 0

  progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}", total=len(train_loader))

  for batch_num, (seq1, seq2) in enumerate(progress_bar):
      (seq1, seq2) = (seq1.to(device), seq2.to(device))
      pred_fwd, pred_bkwd = model(seq1, seq2)
      loss_fwd = loss_fn(pred_fwd.view(-1, pred_fwd.shape[-1]), seq2.view(-1))
      loss_bkwd = loss_fn(pred_bkwd.view(-1, pred_bkwd.shape[-1]), seq1.view(-1))
      loss = loss_fwd+loss_bkwd
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_loss += loss.item()
      progress_bar.set_postfix({'loss': loss.item()})

  epoch_loss /= len(train_loader)
  model.eval()

  with torch.no_grad():
    val_loss = 0
    for batch_num, (seq1, seq2) in enumerate(val_loader):
        (seq1, seq2) = (seq1.to(device), seq2.to(device))
        pred_fwd, pred_bkwd = model(seq1, seq2)
        loss_fwd = loss_fn(pred_fwd.view(-1, pred_fwd.shape[-1]), seq2.view(-1))
        loss_bkwd = loss_fn(pred_bkwd.view(-1, pred_bkwd.shape[-1]), seq1.view(-1))
        val_loss_t = loss_fwd+loss_bkwd
        val_loss += val_loss_t.item()
        
    val_loss /= len(val_loader)
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping....")
        break

  if epoch_num % 1 == 0:
      print(f"Epoch {epoch_num}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")


import pickle
file_path = 'elmo_vocab.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(dataset.vocabulary_words,  f)


