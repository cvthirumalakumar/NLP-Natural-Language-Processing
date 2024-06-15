import re

class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.mails, self.mails_substituted_text = self.replace_mailids(self.text)
        self.urls, self.urls_substituted_text = self.replace_urls(self.mails_substituted_text)
        self.hashtags, self.hashtags_substituted_text = self.replace_hashtags(self.urls_substituted_text)
        self.mentions, self.mentions_substituted_text = self.replace_mentions(self.hashtags_substituted_text)
        self.numbers, self.numbers_substituted_text = self.replace_numbers(self.mentions_substituted_text)
        self.punctuations_handled_text = self.handle_punctuations(self.numbers_substituted_text)
        self.sentence_tokenized_text = self.sentence_tokenization(self.punctuations_handled_text)
        self.tokenized_text = self.word_tokenization(self.sentence_tokenized_text)

    def replace_mailids(self, text):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        return re.findall(email_pattern, text), re.sub(email_pattern, '<MAILID>', text)
    
    def replace_urls(self, text):
        url_pattern = r'\b(?:https?://|www\.)\S+\b'
        return re.findall(url_pattern, text), re.sub(url_pattern, '<URL>', text)

    def replace_hashtags(self, text):
        hashtags_pattern = r'#\w+'
        return re.findall(hashtags_pattern, text), re.sub(hashtags_pattern, '<HASHTAG>', text)
    
    def replace_mentions(self, text):
        mentions_pattern = r'@\w+'
        return re.findall(mentions_pattern, text), re.sub(mentions_pattern, '<MENTION>', text)

    def replace_numbers(self, text):
        numbers_pattern = r'\b\d+\b'
        return re.findall(numbers_pattern, text), re.sub(numbers_pattern, '<NUM>', text)
    
    def handle_punctuations(self,text):
        punctuations_pattern = r'([^\w\s.><\'])'
        text_temp = re.sub(punctuations_pattern, r' \1 ', text)
        text_temp = re.sub(r'\s(\'|")', r' \1 ',text_temp)
        text_temp = re.sub("'", " '",text_temp)
        return text_temp
    
    def sentence_tokenization(self, text):
        text_temp = re.sub(r'\n+',' ',text)
        text_temp = re.sub(r'\s+',' ',text_temp)
        period_split_locs = r'(?<![A-Z])(?<![A-Z][a-z])(?<![A-Z][a-z]s)\.\s?' #(?<![A-Z]) - Abbreavtions/initials, (?<![A-Z][a-z]) - titles like Mr, Dr etc., (?<![A-Z][a-z]s) - Mrs etc
        text_temp = re.sub(period_split_locs,' . ',text_temp)
        text_temp = re.sub(r'\s+',' ',text_temp)
        text_temp = re.split(r'(?<=\s[.?!])\s', text_temp)
        return text_temp
    
    def word_tokenization(self, sentences):
        tokenized_text = []
        for sentence in sentences:
            if len(sentence)!=0:
                tokenized_text.append(re.split(r'\s',sentence.strip().lower()))
        return tokenized_text


    
if __name__ == '__main__':
    text  = input("your text:")
    processed_text = Tokenizer(text).tokenized_text
    print("tokenized text:",processed_text) 