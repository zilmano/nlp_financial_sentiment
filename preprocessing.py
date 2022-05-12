''' Processing functions. Methods to lemattize, do NER and remove ticker names from text'''
from   nltk import word_tokenize
import spacy
import re
import tqdm
from   nltk.stem import WordNetLemmatizer


spacy_model = spacy.load('en_core_web_sm')

def get_ticker_re():
    ticker_re = re.compile("(\(\s*)?\w+\.[A-Z]+(\s*\))?")
    return ticker_re

def get_stop_words():
    punct = list("#$%&'()*+,-./:;=?@[\]^_`{|}~")
    eng_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'to', 'from', 'in', 'on', 'again', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'other', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    remove_list = set(eng_stop_words + punct + ["-DATE-", "said"] )
    return remove_list 

def remove_tickers(ticker_re, header):
    ''' Remove tickers noted ( XXX.X ) or without bracket (XXX.X).'''
    return ticker_re.sub("", header)
        
def lemmatize(header, Lemmatizer):
    return [Lemmatizer.lemmatize(word) for word in header]  

def NER_processing(header, org_name = None):
    ''' 
    Find NER entities using Spacy, and then replace them
    with they labels elcosed in '<>'. For example "John McCullen" will be
    replaced by <PERSON>.
    '''
    spacied_header = spacy_model(header)
    i = 0
    header_sub = []
    for entity in spacied_header.ents:
        header_sub.append(header[i:entity.start_char])
        ## TODO: add making <target_org> tag if the ent.text is equal to 
        ## the name of the compay with regex (org_name). 
        header_sub.append(f" {entity.label_} ")
        i = entity.end_char
    header_sub.append(header[i:])
    header_sub = ''.join(header_sub)
    return header_sub

def tokenize_docs(headers, Lemmatizer, ticker_re):
    '''Process and tokenize.'''
    remove_list = get_stop_words()
    tokenized_headers = []
    for header in tqdm.tqdm(headers):
        header = remove_tickers(ticker_re, header)
        header = NER_processing(header)
        #print(f"({ticker}) {header}")
        header_tokens = [word.lower() for word in word_tokenize(header)
                if word not in remove_list  
                ]
        header_tokens = lemmatize(header_tokens, Lemmatizer)
        tokenized_headers.append(header_tokens)
    
    return tokenized_headers

def preprocess_doc(header):
    preprocess_doc.lemmatizer = WordNetLemmatizer()
    ticker_re = get_ticker_re()
    remove_list = get_stop_words()
    header = remove_tickers(ticker_re, header)
    header = NER_processing(header)
    header_tokens = [word.lower() for word in word_tokenize(header)
                    if word not in remove_list  
                    ]
    header_tokens = lemmatize(header_tokens, preprocess_doc.lemmatizer)
    return ' '.join(header_tokens)
   