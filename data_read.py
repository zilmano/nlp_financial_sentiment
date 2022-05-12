
import pathlib
import re
from nltk import word_tokenize
import pprint
import pandas as pd

import preprocessing

base = pathlib.PosixPath('/Users/ozilman/NLP/finance_sentiment_proj')

###############
##
##  Reading and Processing ReutersNews Dataset
##
##############
def extract_header_from_news_item(file_path, header_re,  date, headers):
    with open(file_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("--") or not line:
                continue
            header_match = header_re.split(line)
            if len(header_match) != 2:
                print(f"Warning: Header line is not as expected. File: {file_path}. \n  Header line: {header_match}")
            else:
                headers.append((date, header_match[1]))
            return

def parse_news_data():
    '''
    iterate over all news documents in subdirectories. Extract the 
    header line from each document with 'extract_header_from_news_item' method.
    '''
    path = base /'financial-news-dataset' / 'ReutersNews106521'
    header_re = re.compile("^.*\(\s*\w+\s*\) - ")
    headers = []
    for day_dir in path.iterdir():
        if not day_dir.is_dir():
            continue
        date = "-".join([day_dir.name[0:4], day_dir.name[4:6], day_dir.name[6:]]) 
        for news_item in day_dir.iterdir():
            try:
                extract_header_from_news_item(news_item, header_re ,date, headers)    
            except Exception as e:
                print(f"  File:{news_item}\n  {e}")

        print(f"Done parsing for date {date}.")  
          
    return headers


def get_sp500_ticker_names():
    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500 = payload[0]
    sp500_history = payload[1] 

    sp500_symbols_and_names  = list(zip(sp500["Symbol"], sp500["Security"]))

    # Get also the tickers that were removed from S&P during and after the news article time span
    sp500_history = sp500_history.iloc[:,[0,3,4]]
    sp500_history.columns = ['date', 'removed_symbol', 'removed_security']
    sp500_history = sp500_history.assign(date = pd.to_datetime(sp500_history['date']))
    removed_tickers = sp500_history[sp500_history['date'] >= '2006-10-20'][['removed_symbol', 'removed_security']].dropna().to_records(index=False)
    removed_tickers = [ticker for ticker in removed_tickers if ticker[0] not in sp500["Symbol"].values]
    sp500_symbols_and_names.extend(removed_tickers)

    # Make a mapping of ticker symbols to security names
    sp500_symbols_to_names = dict(sp500_symbols_and_names)
    return sp500_symbols_to_names


def get_relevant_news(symbols, security_names, headers):
    mentions_cnt = 0
    relevant_news = []
    for date, header in headers:
        header_tokenized = word_tokenize(header)
        if header_tokenized[0] == 'A':
            header_tokenized = header_tokenized[1:]
        for symbol in symbols:
            if symbol in header_tokenized or security_names[symbol] in header:
                #print(f"Symbol: {symbol} -> Header: {header}")
                mentions_cnt += 1
                relevant_news.append([date, symbol, 0, " ".join(header_tokenized)])

    print(f"Relvant news #: {mentions_cnt}")
    return relevant_news


#############
##
## Reading and processing FinancialPhraseBank
##
#############  

def preprocess_doc(header, lemmatizer):
    ticker_re = preprocessing.get_ticker_re()
    remove_list = preprocessing.get_stop_words()
    header = preprocessing.remove_tickers(ticker_re, header)
    header = preprocessing.NER_processing(header)
    header_tokens = [word.lower() for word in word_tokenize(header)
                    if word not in remove_list  
                    ]
    header_tokens = preprocessing.lemmatize(header_tokens, lemmatizer)
    return ' '.join(header_tokens)


def load_fin_pharsebank():
    path    = base /'FinancialPhraseBank-v1.0' / 'Sentences_66Agree.txt'
    phrases = []
    labels  = []
    euro_re = re.compile('(EURO?|euro?)\s*\d+\s*([.,]\s*\d+\s*)(mn|m)?\s+')
    neg_percent_re = re.compile('-\d+\s*%')
    with open(path, 'r', encoding='latin-1') as fh:
        for line in fh:
            line = line.strip()
            phrase, sentiment = line.split('@')
            phrase = euro_re.sub(' 127 million dollars ',phrase)
            phrase = neg_percent_re.sub( ' percent ',phrase)
            if sentiment == "positive":
                phrases.append(phrase)
                labels.append(1)
            elif sentiment == "negative": 
                phrases.append(phrase)
                labels.append(0)
    return phrases, labels