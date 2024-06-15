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
import random
from sklearn.metrics import accuracy_score
import glob
from sklearn.metrics import classification_report, confusion_matrix
import pickle

random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


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


train_text, train_labels_original = read_csv('ANLP-2/train.csv')
test_text, test_labels_original = read_csv('ANLP-2/test.csv')

train_labels = [label-1 for label in train_labels_original]
test_labels = [label-1 for label in test_labels_original]

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

class LSTMDataset(Dataset):
  def __init__(self, text, labels, vocabulary_words:Vocab|None=None):
    """Initialize the dataset. Setup Code goes here"""
    self.labels = labels
    self.sentences = []
    for sample in text:
        t = []
        for temp_sent in tokenize(sample):
            t += temp_sent
        self.sentences.append(t)

    if vocabulary_words is None:
      self.vocabulary_words = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN])
      self.vocabulary_words.set_default_index(self.vocabulary_words[UNKNOWN_TOKEN])
    else:
      self.vocabulary_words = vocabulary_words

   

  def __len__(self) -> int:
    """Returns number of datapoints."""
    return len(self.sentences)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the datapoint at `index`."""
    return torch.tensor(self.vocabulary_words.lookup_indices(self.sentences[index])), torch.tensor(self.labels[index])

  def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a list of datapoints, batch them together"""
    sentences = [i[0] for i in batch]
    labels = [i[1] for i in batch]
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary_words[PAD_TOKEN])
    return padded_sentences, torch.tensor(labels)


file_path = 'elmo_vocab.pkl'
with open(file_path, 'rb') as f:
    elmo_vocab = pickle.load(f)


dataset = LSTMDataset(train_text, train_labels, vocabulary_words=elmo_vocab)
test_dataset = LSTMDataset(test_text, test_labels, vocabulary_words=elmo_vocab)

train_size, val_size = 0.8, 0.2
batch_size = 64
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)

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
            return (embeddings_concat, lstm_layer1, lstm_layer2)
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

def train(model, model_name, n_epochs=25):
  val_accuracies_epoch_wise = []
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # device =  "cpu"
  print("Utilising",device)
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  model = model.to(device)
  model_path = "models/elmo-classifier-"+model_name+".pt"
  early_stopping = EarlyStopping(patience=5, path=model_path)

  for epoch_num in range(n_epochs):
      model.train()
      epoch_loss = 0

      progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}", total=len(train_loader))

      for batch_num, (input, labels) in enumerate(progress_bar):
          (input, labels) = (input.to(device), labels.to(device))
          optimizer.zero_grad()
          pred = model(input)
          loss = loss_fn(pred, labels)
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item()
          progress_bar.set_postfix({'loss': loss.item()})

      epoch_loss /= len(train_loader)
      model.eval()

      with torch.no_grad():
        val_loss = 0
        for batch_num, (input, labels) in enumerate(val_loader):
            (input, labels) = (input.to(device), labels.to(device))
            pred = model(input)
            val_loss += loss_fn(pred, labels)
        val_loss /= len(val_loader)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping....")
            break

      if epoch_num % 1 == 0:
          print(f"Epoch {epoch_num}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
  print("best val loss is",early_stopping.best_score)
  return val_accuracies_epoch_wise

def test(model, model_name):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("Utilising",device)
  # device =  "cpu"
  model = model.to(device)
  model_path = "models/elmo-classifier-"+model_name+".pt"
  state_dict = torch.load(model_path)
  model.load_state_dict(state_dict)
  model.eval()

  with torch.no_grad():
    all_predictions = []
    all_ground_truth = []
    for batch_num, (input, labels) in tqdm(enumerate(val_loader)):
        (input, labels) = (input.to(device), labels.to(device))
        pred = model(input)
        predictions = torch.argmax(pred, dim=1).cpu().numpy()
        ground_truth = labels.cpu().numpy()
        all_predictions += list(predictions)
        all_ground_truth += list(ground_truth)

    accuracy_val = accuracy_score(all_ground_truth, all_predictions)

    all_predictions = []
    all_ground_truth = []
    for batch_num, (input, labels) in tqdm(enumerate(test_loader)):
        (input, labels) = (input.to(device), labels.to(device))
        pred = model(input)
        predictions = torch.argmax(pred, dim=1).cpu().numpy()
        ground_truth = labels.cpu().numpy()
        all_predictions.append(predictions)
        all_ground_truth.append(ground_truth)
    all_predictions = np.concatenate(all_predictions)
    all_ground_truth = np.concatenate(all_ground_truth)

    accuracy_test = accuracy_score(all_ground_truth, all_predictions)
    report = classification_report(all_ground_truth, all_predictions)
    print("Classification Report")
    print(report)
    conf_matrix = confusion_matrix(all_ground_truth, all_predictions)
    print("Confusion Matrix")
    print(conf_matrix)
    return accuracy_val, accuracy_test
  
elmo_state_dict = torch.load('models/Elmo.pt')
elmo = Elmo(len(dataset.vocabulary_words), 256, return_embeddings=True)
elmo.to('cuda')
elmo.load_state_dict(elmo_state_dict)
for param in elmo.parameters():
    param.requires_grad = False


# Trainable lambdas
class Elmo_Classifier1(torch.nn.Module):
    def __init__(self, elmo, op_dim, n_layers=3, hidden_dim=256, embedding_dim=512):
        super(Elmo_Classifier1, self).__init__()
        self.elmo = elmo
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, op_dim)
        self.lambda1 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.lambda2 = nn.Parameter(torch.randn(1), requires_grad=True)
        self.lambda3 = nn.Parameter(torch.randn(1), requires_grad=True)
        
    def forward(self, x):
        embeddings_concat, lstm_layer1, lstm_layer2 = self.elmo(x, x)
        out = self.lambda1 * embeddings_concat + self.lambda2 * lstm_layer1 + self.lambda3 * lstm_layer2
        out, _ = self.rnn(out)
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out
    
model_name =  '1'
model = Elmo_Classifier1(elmo, 4)
val_accuracy_epoch_wise = train(model, model_name, n_epochs=50)
print("Testing")
accuracy_val, accuracy_test = test(model, model_name)
print(f"Model-{model_name}:{accuracy_val}:{accuracy_test}")


# Frozen Lambdas
class Elmo_Classifier2(torch.nn.Module):
    def __init__(self, elmo, op_dim, n_layers=3, hidden_dim=256, embedding_dim=512):
        super(Elmo_Classifier2, self).__init__()
        self.elmo = elmo
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, op_dim)
        self.lambda1 = torch.rand(())
        self.lambda2 = torch.rand(())
        self.lambda3 = torch.rand(())
        
    def forward(self, x):
        embeddings_concat, lstm_layer1, lstm_layer2 = self.elmo(x, x)
        out = self.lambda1 * embeddings_concat + self.lambda2 * lstm_layer1 + self.lambda3 * lstm_layer2
        # print(out.shape)
        out, _ = self.rnn(out)
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out
    
model_name =  '2'
model = Elmo_Classifier2(elmo, 4)
val_accuracy_epoch_wise = train(model, model_name, n_epochs=50)
print("Testing")
accuracy_val, accuracy_test = test(model, model_name)
print(f"Model-{model_name}:{accuracy_val}:{accuracy_test}")


# Function
class Elmo_Classifier3(torch.nn.Module):
    def __init__(self, elmo, op_dim, n_layers=3, hidden_dim=256, embedding_dim=512):
        super(Elmo_Classifier3, self).__init__()
        self.elmo = elmo
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_1 = nn.Linear(embedding_dim*3, embedding_dim)
        self.fc = nn.Linear(hidden_dim, op_dim)

        
    def forward(self, x):
        embeddings_concat, lstm_layer1, lstm_layer2 = self.elmo(x, x)
        out = torch.cat((embeddings_concat, lstm_layer1, lstm_layer2), dim=-1)
        out = self.fc_1(out)
        # print(out.shape)
        out, _ = self.rnn(out)
        out = torch.mean(out, dim=1)
        out = self.fc(out)
        return out
    

model_name =  '3'
model = Elmo_Classifier3(elmo, 4)
val_accuracy_epoch_wise = train(model, model_name, n_epochs=50)
print("Testing")
accuracy_val, accuracy_test = test(model, model_name)
print(f"Model-{model_name}:{accuracy_val}:{accuracy_test}")