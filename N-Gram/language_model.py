import sys 
import os
from tokenizer import Tokenizer
import re
from scipy.stats import linregress
import numpy as np
import random
import math
import pickle
np.random.seed(42)

class Ngram():
    def __init__(self, N=3, smoothing = None, remove_punctuations = True):
        self.N = N
        self.smoothing = smoothing
        self.remove_punctuations = remove_punctuations
        self.counts_dict = None

    def punctuation_remover(self, text):
        '''removes punctuation only tokens and also punctuations within tokens'''
        new_text = []
        for sentence in text:
            new_sentence = []
            for word in sentence:
                word = re.sub(r'([^\w\s><\']|_)','',word).strip()
                if len(word) != 0:
                    new_sentence.append(word)
            new_text.append(new_sentence)
        return new_text

    def sentence_to_ngram(self, sentence, n):
        '''Takes sentence(list of tokens) and n(order) as input and retuns ngrams.'''
        n_grams = []
        for i in range(len(sentence)-n+1):
            n_gram = " ".join(sentence[i:i+n])
            n_grams.append(n_gram)
        return n_grams
        
    def ngram_counter(self, tokenized_text, n):
        '''Takes tokenized_text(list of list of tokens[for each sentence]) and n(order) as input and returns ngram counts.'''
        counts_dict = {}
        for sentence in tokenized_text:
            for n_gram in self.sentence_to_ngram(sentence,n):
                if n_gram in counts_dict:
                    counts_dict[n_gram] += 1
                else:
                    counts_dict[n_gram] = 1
        return counts_dict

    def train(self, corpus_path):
        '''Takes N(order) and corpus_path as input and return ngram counts from n = 1:N'''
        self.corpus_path = corpus_path
        f = open(corpus_path,'r',encoding='utf8')
        text = f.read()
        f.close()
        tokenized_text = Tokenizer(text).tokenized_text
        random.shuffle(tokenized_text)
        self.train_set, self.test_set = tokenized_text[:-1000], tokenized_text[-1000:]
        
        if self.remove_punctuations:
            self.train_set = self.punctuation_remover(self.train_set)
            self.test_set = self.punctuation_remover(self.test_set)
            
        self.train_set = [['<START>' for _ in range(self.N-1)] + sentence + ['<END>'] for sentence in self.train_set]
        ngram_counts = {}
        for n in range(1,self.N+1):
            ngram_counts[n] = self.ngram_counter(self.train_set,n)
        self.counts_dict = ngram_counts
        self.LinearInterpolation()
        self.GoodTuring()
    
    def GoodTuring(self):
        '''Creates r-Nr dictionary and fit a line for log(r)-log(Nr)'''
        nr = {}
        for n_gram in self.counts_dict[self.N]:
            r = self.counts_dict[self.N][n_gram]
            if r in nr:
                nr[r] += 1
            else:
                nr[r] = 1
        self.nr = dict(sorted(nr.items(), key=lambda item: item[1]))
        self.lr = linregress(np.log(list(lm.nr.keys())),np.log(list(lm.nr.values())))
        # re-estimating counts using good turing method
        self.gt_probs={}
        for n_gram in self.counts_dict[self.N]:
            r = self.counts_dict[self.N][n_gram]
            r_1 = r+1
            nr = self.nr[r] if r<20 else np.exp(self.lr.slope*np.log(r)+self.lr.intercept)
            nr_1 = self.nr[r_1] if r_1<20 else np.exp(self.lr.slope*np.log(r_1)+self.lr.intercept)
            r_star = ((r_1)*nr_1)/nr
            self.gt_probs[n_gram] = r_star
        self.total_n = sum(self.counts_dict[self.N].values())
        tottal_ngrams_possible = np.power(len(self.counts_dict[1]),self.N)
        self.gt_n0 = self.nr[1]/(tottal_ngrams_possible-len(self.counts_dict[self.N]))
            
    def LinearInterpolation(self):
        '''Estimates lambdas'''
        lambdas_t = [0 for _ in range(self.N)]
        n_tokens = sum(self.counts_dict[1].values())
        for n_gram in self.counts_dict[self.N]:
            words = n_gram.split()
            expression_values = []
            freqs = []
            for n_temp in range(1, self.N+1):
                if n_temp == 1:
                    expression_values.append((self.counts_dict[1][words[-1]]-1)/(n_tokens-1))
                    freqs.append(self.counts_dict[1][words[-1]])
                else:
                    n_gram_temp = " ".join(words[-n_temp:])
                    n_gram_temp_lower = " ".join(n_gram_temp.split()[:-1])
                    value = (self.counts_dict[n_temp][n_gram_temp]-1)/(self.counts_dict[n_temp-1][n_gram_temp_lower]-1) if self.counts_dict[n_temp-1][n_gram_temp_lower]!=1 else 0
                    expression_values.append(value)
                    freqs.append(self.counts_dict[n_temp][n_gram_temp])
                    
            max_n = expression_values.index(max(expression_values))
            lambdas_t[max_n] += freqs[-1]
        sum_lambdas = sum(lambdas_t)
        self.lambdas = [lambda_v/sum_lambdas for lambda_v in lambdas_t]
        
    def score_ngram(self, n_gram):
        '''Returns probability of given Ngram based on the smoothing option'''
        if self.smoothing == None:
        # No smoothing
            context = " ".join(n_gram.split()[:-1])
            if n_gram in self.counts_dict[self.N]:
                if context in self.counts_dict[self.N-1]:
                    c_ngram = self.counts_dict[self.N][n_gram]
                    c_context = self.counts_dict[self.N-1][context]
                    return c_ngram/c_context
                else:
                    return 10e-5
            else:
                return 10e-5
        elif self.smoothing == 'i':
        #Linear interpolation scoring
            score_i = 0
            words = n_gram.split()
            for n_t in range(1, self.N+1):
                n_gram_t = " ".join(words[-n_t:])
                if n_t == 1:
                    if n_gram_t in self.counts_dict[1]:
                        score_i += self.lambdas[0]*(self.counts_dict[1][n_gram_t]/sum(self.counts_dict[1].values()))
                    else:
                        score_i += self.lambdas[0]*10e-5
                else:
                    context = " ".join(n_gram_t.split()[:-1])
                    if n_gram_t in self.counts_dict[n_t]:
                        score_i += self.lambdas[n_t-1]*(self.counts_dict[n_t][n_gram_t]/self.counts_dict[n_t-1][context])
            return score_i

        elif self.smoothing == 'g':
            context = " ".join(n_gram.split()[:-1])
            c_star_den = 0
            c_star_num = self.gt_probs[n_gram] if n_gram in self.gt_probs else self.gt_n0
            for word in self.counts_dict[1]:
                n_gram_temp = " ".join([context,word])
                c_star_den += self.gt_probs[n_gram_temp] if n_gram_temp in self.gt_probs else self.gt_n0
            return c_star_num/c_star_den

            
    def score(self, sentence, return_probs = False):
        '''Takes LM(ngram_counts), N(order), sentence(tokenized) as input and return score'''
        tokenized_text = Tokenizer(sentence).tokenized_text
        if self.remove_punctuations:
            tokenized_text = self.punctuation_remover(tokenized_text)
        tokenized_text = ['<START>' for _ in range(self.N-1)] + tokenized_text[0] + ['<END>']
        scores = []
        for n_gram in self.sentence_to_ngram(tokenized_text, self.N):
            scores.append(self.score_ngram(n_gram))
            
        if return_probs == True:
            return sum(np.log(scores)), scores
        else:
            return sum(np.log(scores))

    def perplexity(self, sentence):
        '''returns perplexity of a given sentence'''
        score, probs = self.score(sentence, return_probs=True)
        return np.power(2,-1*np.mean(np.log2(probs)))


    def generate(self,sentence, top_n = 3):
        '''returns top n probable words for next word given context'''
        tokenized_text = Tokenizer(sentence).tokenized_text
        if self.remove_punctuations:
            tokenized_text = self.punctuation_remover(tokenized_text)
        tokenized_text = ['<START>' for _ in range(self.N-1)] + tokenized_text[0]
        n_grams = self.sentence_to_ngram(tokenized_text, self.N)
        context = " ".join(n_grams[-1].split()[1:])
        words, scores = [], []
        for word in self.counts_dict[1]:
            if word != '<START>':
                n_gram_t = " ".join([context,word])
                scores.append(self.score_ngram(n_gram_t))
                words.append(word)
        sorted_indices = np.argsort(scores)[::-1]
        return {words[i]: scores[i] for i in sorted_indices[:top_n]}

    def evaluate(self):
        original_smoothing = self.smoothing
        self.test_set = [['<START>' for _ in range(self.N-1)] + sentence + ['<END>'] for sentence in self.test_set]
        for smoothing_option in ['i','g']:
            self.smoothing = smoothing_option
            self.test_set_perplexity = {}
            for sentence in self.test_set:
                scores = []
                for n_gram in self.sentence_to_ngram(sentence, self.N):
                    scores.append(self.score_ngram(n_gram))
                sentence = " ".join(sentence)
                self.test_set_perplexity[sentence] = np.power(2,-1*np.mean(np.log2(scores)))
    
            self.train_set_perplexity = {}
            for sentence in self.train_set:
                scores = []
                for n_gram in self.sentence_to_ngram(sentence, self.N):
                    scores.append(self.score_ngram(n_gram))
                sentence = " ".join(sentence)
                self.train_set_perplexity[sentence] = np.power(2,-1*np.mean(np.log2(scores)))
    
    
            file_name = self.corpus_path.split("/")[-1].split(".")[0]+"_"+str(self.N)+"_"+self.smoothing+"_train.txt"
            f_train = open(file_name, 'w', encoding='utf8')
            f_train.write(str(sum(self.train_set_perplexity.values())/len(self.train_set_perplexity))+"\n")
            for sentence in self.train_set_perplexity:
                f_train.write(sentence+"\t"+str(self.train_set_perplexity[sentence])+"\n")
            f_train.close()
    
            file_name = self.corpus_path.split("/")[-1].split(".")[0]+"_"+str(self.N)+"_"+self.smoothing+"_test.txt"
            f_test = open(file_name, 'w', encoding='utf8')
            f_test.write(str(sum(self.test_set_perplexity.values())/len(self.test_set_perplexity))+"\n")
            for sentence in self.test_set_perplexity:
                f_test.write(sentence+"\t"+str(self.test_set_perplexity[sentence])+"\n")
            f_test.close()
        self.smoothing = original_smoothing
    
    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump([self.counts_dict, self.lambdas, self.nr, self.lr, self.gt_n0, self.gt_probs], f)
    
    def load(self, file_name):
        with open(file_name, 'rb') as f:
                self.counts_dict, self.lambdas, self.nr, self.lr, self.gt_n0, self.gt_probs = pickle.load(f)
        
if __name__ == '__main__':
    if len(sys.argv) == 3:
        if sys.argv[1] not in ['g','i','n']:
            print("Please enter a valid first argument as g (for Good-Turing Smoothing) or i (for Linear Interpolation) or n (for no smoothing)")
            exit()
        if not os.path.exists(sys.argv[2]):
            print("Please enter a valid corpus path. {} does not exists".format(sys.argv[2]))
            exit()
    else:
        print('''Please provide arguments for smoothing and corpus path..
        language_mode.py <smoothing> <corpus_path>
        eg. language_model.py i corpus.txt''')
        exit()

    
    '''
    #following is the code is used to train LMs
    lm = Ngram(N=<order>, smoothing=<smoothing_option>)
    lm.train(<corpus_path>)
    '''
    if sys.argv[1] == 'n':
        smoothing = None
    else:
        smoothing = sys.argv[1]
    
    if 'Pride_and_Prejudice-Jane_Austen' in sys.argv[2]:
        lm_file = 'lm_Pride_and_Prejudice-Jane_Austen.pkl'
    elif 'Ulysses_James_Joyce' in sys.argv[2]:
        lm_file = 'lm_Ulysses_James_Joyce.pkl'


    lm = Ngram(3, smoothing=smoothing)
    lm.load(lm_file)
    text  = input("Input sentence:")
    print("Log likelihood score: ",lm.score(text))
    # print("Perplexity: ",lm.perplexity(text))