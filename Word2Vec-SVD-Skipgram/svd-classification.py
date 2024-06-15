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
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

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
    tokenized_text = [['<start>']+word_tokenize(t)+['<end>'] for t in new_sents] 
    return tokenized_text

class News_Dataset(Dataset):
    def __init__(self, text, labels, word2idx, embeddings):
        self.labels = labels
        self.word2idx = word2idx
        self.embeddings = embeddings
        self.processed_text = []
        for sample in text:
            t = []
            for temp_sent in tokenize(sample):
                t += temp_sent
            indices = [word2idx[token] if token in word2idx else word2idx['<unk>'] for token in t]
            self.processed_text.append(indices)

    def __len__(self):
        return len(self.processed_text)
    
    def __getitem__(self, idx):
        sample = self.embeddings[self.processed_text[idx]]
        label = self.labels[idx]
        return sample, torch.tensor(label)

    def collate(self, batch):
        sentences = [i[0] for i in batch]
        lengths = [len(sent) for sent in sentences]
        labels = [i[1] for i in batch]
        padded_sentences = pad_sequence(sentences, batch_first=True) 
        return padded_sentences, torch.tensor(lengths, dtype=torch.int64), torch.tensor(labels)
    

class RNNModel(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_size, num_layers, bidirectionality=False, embedding_dim=256):
        super(RNNModel, self).__init__()
        self.bidirectionality = bidirectionality
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=bidirectionality)
        if self.bidirectionality:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        out = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        last_valid_time_step = (lengths - 1).unsqueeze(1).expand(-1, out.size(2)).unsqueeze(1)
        last_valid_time_step = last_valid_time_step.to('cuda')
        out = torch.gather(out, 1, last_valid_time_step).squeeze(1)
        # out = out[:, -1] 
        out = self.fc(out)
        return out
        
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
  model_path = "svd_classification/svd-classification-"+model_name+".pt"
  early_stopping = EarlyStopping(patience=5, path=model_path)

  for epoch_num in range(n_epochs):
      model.train()
      epoch_loss = 0

      progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}", total=len(train_loader))

      for batch_num, (input, lengths, labels) in enumerate(progress_bar):
          (input, labels) = (input.to(device), labels.to(device))
          optimizer.zero_grad()
          pred = model(input.float(), lengths)
          loss = loss_fn(pred, labels)
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item()
          progress_bar.set_postfix({'loss': loss.item()})

      epoch_loss /= len(train_loader)
      model.eval()

      with torch.no_grad():
        val_loss = 0
        for batch_num, (input, lengths, labels) in enumerate(val_loader):
            (input, labels) = (input.to(device), labels.to(device))
            pred = model(input.float(), lengths)
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
  model_path = "svd_classification/svd-classification-"+model_name+".pt"
  state_dict = torch.load(model_path)
  model.load_state_dict(state_dict)
  model.eval()

  with torch.no_grad():
    all_predictions = []
    all_ground_truth = []
    for batch_num, (input, lengths, labels) in tqdm(enumerate(val_loader)):
        (input, labels) = (input.to(device), labels.to(device))
        pred = model(input.float(), lengths)
        predictions = torch.argmax(pred, dim=1).cpu().numpy()
        ground_truth = labels.cpu().numpy()
        all_predictions += list(predictions)
        all_ground_truth += list(ground_truth)

    accuracy_val = accuracy_score(all_ground_truth, all_predictions)

    all_predictions = []
    all_ground_truth = []
    for batch_num, (input, lengths, labels) in tqdm(enumerate(test_loader)):
        (input, labels) = (input.to(device), labels.to(device))
        pred = model(input.float(), lengths)
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
  

  
if __name__ == '__main__':

    print("Reading files")
    train_text, train_labels_original = read_csv('ANLP-2/train.csv')
    test_text, test_labels_original = read_csv('ANLP-2/test.csv')

    train_labels = [label-1 for label in train_labels_original]
    test_labels = [label-1 for label in test_labels_original]

    model_configs_lstm = {
        'model_1':{
            'hidden_layers':3,
            'hidden_dim':256,
            'embedding_size':256,
            'bidirectionality':False
        }
    }

    val_accuracies = []
    test_accuracies = []
    for file in glob.glob("svd/*"):
        window_size = file.split("_")[-1].split(".")[0]
        print(f"Window size {window_size}")
        print("Loading word vectros")
        word2idx,word_vectors_tensor_t =  torch.load(file)
        print("Creating Train Dataset")
        dataset = News_Dataset(train_text, train_labels, word2idx, word_vectors_tensor_t)
        print("Creating Test Dataset")
        test_dataset = News_Dataset(test_text, test_labels, word2idx, word_vectors_tensor_t)
        
        print("Splitting and creating data loaders")
        train_size, val_size = 0.8, 0.2
        batch_size = 128
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        print("Training")
        model_name =  'model_1'
        hidden_dim = model_configs_lstm[model_name]['hidden_dim']
        hidden_layers = model_configs_lstm[model_name]['hidden_layers']
        embedding_dim = model_configs_lstm[model_name]['embedding_size']
        bidirectionality = model_configs_lstm[model_name]['bidirectionality']
        model = RNNModel(len(word2idx), 4, hidden_dim, hidden_layers,bidirectionality=bidirectionality, embedding_dim=256)
        val_accuracy_epoch_wise = train(model, model_name+"-"+window_size, n_epochs=50)
        print("Testing")
        accuracy_val, accuracy_test = test(model, model_name+"-"+window_size)
        print(f"Model-{model_name}-{window_size}:{accuracy_val}:{accuracy_test}")
        val_accuracies.append(accuracy_val*100)
        test_accuracies.append(accuracy_test*100)

    


    
        