import torch,torchtext
import sys
import pickle

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_size, num_layers, bidirectionality=False, embedding_dim=256):
        super(LSTMModel, self).__init__()
        self.bidirectionality = bidirectionality
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectionality)
        if self.bidirectionality:
          self.fc = torch.nn.Linear(hidden_size*2, num_classes)
        else:
          self.fc = torch.nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        embedded = self.embedding(x)  
        out, _ = self.lstm(embedded)  
        out = self.fc(out)  
        return out

class FFNN(torch.nn.Module):
    def __init__(self, vocab_size, output_dim, context_size, hidden_dims, activation='relu', embedding_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layers = torch.nn.ModuleList()
        prev_dim = embedding_dim*context_size
        for dim in hidden_dims:
            self.hidden_layers.append(torch.nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.output_layer = torch.nn.Linear(prev_dim, output_dim)

        if activation == 'relu':
            self.activation_function = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation_function = torch.nn.Sigmoid()
        elif activation == 'tanh':
            self.activation_function = torch.nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        return x
    
model_configs_fnn = {
    'Model_1':{
        'hidden_dims':[512, 256, 128, 64],
        'activation':'relu',
        'embedding_size':256
    },
    'Model_2':{
        'hidden_dims':[256, 256, 128, 128],
        'activation':'tanh',
        'embedding_size':128
    },
    'Model_3':{
        'hidden_dims':[512, 512, 256, 128, 64],
        'activation':'relu',
        'embedding_size':256
    }
}

model_configs_lstm = {
    'model_1':{
        'hidden_layers':10,
        'hidden_dim':256,
        'embedding_size':128,
        'bidirectionality':True
    },
    'model_2':{
        'hidden_layers':7,
        'hidden_dim':256,
        'embedding_size':256,
        'bidirectionality':False
        },
    'model_3':{
        'hidden_layers':5,
        'hidden_dim':512,
        'embedding_size':512,
        'bidirectionality':True
    }
}

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('please give parameter as -f (for FFNN) or -r (for LSTM)\n eg..python pos_tagger.py -f')
        sys.exit(1)
    model_type = sys.argv[1] 
    if model_type not in ['-f','-r']:
        print('please give parameter as -f (for FFNN) or -r (for LSTM)\n eg..python pos_tagger.py -f')
        sys.exit(1)


    sentence = input("Input Sentence:").lower()

    START_TOKEN = "<s>"
    END_TOKEN = "</s>"
    if model_type == '-f':
        ''' COde for inferencing FFNN'''
        p = s = 1
        context_size = (p*2)+1
        words = []
        original_sent = sentence
        sentence = [START_TOKEN]*p + sentence.split() + [END_TOKEN]*s
        for i in range (p, len(sentence)-s):
            start_idx, end_idx = i-p, i+s+1
            word_with_context = sentence[start_idx:end_idx]
            words.append(word_with_context)

        with open('./ffnn_vocab.pkl', "rb") as f:
            vocabulary_words, vocabulary_tags = pickle.load(f)
            
        ip_indices = []
        for context in words:
            word_indices = [vocabulary_words[token] for token in context]
            ip_indices.append(word_indices)
        ip_indices = torch.tensor(ip_indices)

        model_name = 'Model_1'
        hidden_dim = model_configs_fnn[model_name]['hidden_dims']
        activation = model_configs_fnn[model_name]['activation']
        embedding_dim = model_configs_fnn[model_name]['embedding_size']

        model = FFNN(len(vocabulary_words), len(vocabulary_tags), context_size,
                 hidden_dim, activation, embedding_dim=embedding_dim)
        
        model_path = "best_ffnn.pt"
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            pred = model(ip_indices)
            predictions = torch.argmax(pred, axis=-1).cpu().numpy().flatten()
            labels = [vocabulary_tags.lookup_token(prediction) for prediction in predictions]
        
        words = original_sent.split()
        for i in range(len(words)):
            print(words[i]+" "+labels[i])
        
    if model_type == '-r':
        '''Code for interencing LSTM'''
        with open('./lstm_vocab.pkl', "rb") as f:
            vocabulary_words, vocabulary_tags = pickle.load(f)
        ip_indices = torch.tensor(vocabulary_words.lookup_indices(sentence.split()))

        model_name = 'model_3'
        hidden_dim = model_configs_lstm[model_name]['hidden_dim']
        hidden_layers = model_configs_lstm[model_name]['hidden_layers']
        embedding_dim = model_configs_lstm[model_name]['embedding_size']
        bidirectionality = model_configs_lstm[model_name]['bidirectionality']

        model = LSTMModel(len(vocabulary_words), len(vocabulary_tags), hidden_dim, hidden_layers,bidirectionality=bidirectionality, embedding_dim=embedding_dim)
        model_path = "best_lstm.pt"
        state_dict = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        ip_indices = ip_indices.unsqueeze(0)
        print(ip_indices.shape)
        with torch.no_grad():
            pred = model(ip_indices)
            predictions = torch.argmax(pred, axis=-1).cpu().numpy().flatten()
            labels = [vocabulary_tags.lookup_token(prediction) for prediction in predictions]
        
        words = sentence.split()
        for i in range(len(words)):
            print(words[i]+" "+labels[i])

 






