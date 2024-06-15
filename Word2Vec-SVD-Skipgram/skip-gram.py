import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import contractions
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    tokenized_text = [['<start>']+word_tokenize(t)+['<end>'] for t in new_sents] 
    return tokenized_text
    
def word_frequency(text):
    word_dict={}
    for sample in text:
        for sentence in sample:
            for word in sentence:
                if word in word_dict:
                    word_dict[word]+=1
                else:
                    word_dict[word]=1
    return word_dict

def replace_low_frequency_words_with_unk(text,cut_off_frequency=3):
    word_dict = word_frequency(text)
    unk_words = set([word for word, frequency in word_dict.items() if frequency <= cut_off_frequency])
    out_text = []
    for sample in text:
        replaced_sentences = [[token if token not in unk_words else '<unk>' for token in sentence] for sentence in sample]
        out_text += replaced_sentences
    words = []
    for sentence in out_text:
        words += sentence
    return out_text, unk_words, set(words)

def get_word2idx(words):
    word2idx = {}
    for i,word in enumerate(words):
        word2idx[word] = i
    return word2idx

class SkipGramDataset(Dataset):
    def __init__(self, sentences, word2idx, window_size=2, num_negative_samples=5):
        self.sentences = sentences
        self.vocab_size = len(word2idx)
        self.num_negative_samples = num_negative_samples
        self.word2idx = word2idx

        self.postitve_pairs = []
        for sentence in sentences:
            for i, word in enumerate(sentence):
                center_word_idx = word2idx.get(word, None)
                start_index = max(0, i - window_size)
                end_index = min(len(sentence), i+window_size+1)
                context_words = sentence[start_index:i] + sentence[i+1:end_index]
                
                for context_word in context_words:
                    context_word_idx = word2idx[context_word]
                    if context_word_idx is not None:
                        self.postitve_pairs.append((center_word_idx, context_word_idx))
    
    def __len__(self):
        return len(self.postitve_pairs)
    
    def __getitem__(self, idx):
        target, context = self.postitve_pairs[idx]
        negative_samples = np.random.choice(self.vocab_size, size=self.num_negative_samples)
        return torch.tensor(target), torch.tensor(context), torch.tensor(negative_samples)
    
class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256):
        super(SkipGramNegativeSampling, self).__init__()
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, target, context, negative_samples):
        target_embed = self.target_embedding(target)
        context_embed = self.context_embedding(context)
        neg_samples_embed = self.context_embedding(negative_samples)
        pos_score = torch.sum(target_embed * context_embed, dim=1)
        neg_score = torch.bmm(target_embed.unsqueeze(1),neg_samples_embed.transpose(1,2)).squeeze()
        
        return pos_score, neg_score
    
if __name__ == '__main__':
        
    print("Reading files")
    train_text, train_labels = read_csv('ANLP-2/train.csv')
    test_text, test_labels = read_csv('ANLP-2/test.csv')
    print("Tokenizing....")
    tokenised_text = [tokenize(sample) for sample in train_text]
    print("Replacing low freq words with unk....")
    tokenized_text_unk, unk_words, uniq_words = replace_low_frequency_words_with_unk(tokenised_text)
    word2idx = get_word2idx(uniq_words)

    for window_size in [1,2,3,4,5]:
        print(f"window size {window_size}")
        print("Creating dataset")
        batch_size = 30000
        dataset = SkipGramDataset(tokenized_text_unk, word2idx, window_size=window_size, num_negative_samples=5)
        print("Creating dataloader")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("Training started")
        load_from_previous_check_points = False
        epochs = 5
        model = SkipGramNegativeSampling(len(word2idx), embedding_dim=256)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
               
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"utilising {device}")
        model = model.to(device)
        
        if load_from_previous_check_points:
            print("Loading model")
            checkpoint = torch.load('skipgram_checkpoints/model_checkpoint_skipgram_'+str(window_size)+'.pth',map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
        for epoch in range(epochs):
            total_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", total=len(dataloader))
            for batch_num, (target, context, negative_samples) in enumerate(progress_bar):
                optimizer.zero_grad()
                target, context, negative_samples = target.to(device), context.to(device), negative_samples.to(device)
                pos_score, neg_score = model(target, context, negative_samples)
                
                pos_labels = torch.ones_like(pos_score)
                neg_labels = torch.zeros_like(neg_score)
                loss = loss_fn(pos_score, pos_labels) + loss_fn(neg_score, neg_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

                if batch_num%50==0:
                    torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    },'skipgram_checkpoints/model_checkpoint_skipgram_'+str(window_size)+'.pth')
            total_loss /= len(dataloader)
        
            print(f"Epoch {epoch}, Train Loss: {total_loss}")
            print("Saving model checkpoints")
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            },'skipgram_checkpoints/model_checkpoint_skipgram_'+str(window_size)+'.pth')

        embd_file_name = "skipgram/skipgram-word-vector-"+str(window_size)+".pt"
        print(f"saving embeddings as {embd_file_name}")
        target_embeddings = model.target_embedding.weight.detach()
        context_embeddings = model.context_embedding.weight.detach()
        embeddings_dict = {
            'target_embeddings': target_embeddings,
            'context_embeddings': context_embeddings
        }
        torch.save(embeddings_dict,embd_file_name)
