"""
Created on Fri Oct 15 17:08:53 2021

@author: hatem
"""
# Importing libraries
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import pandas as pd
from lxml import *
import scrapy
import lxml.html
from scrapy.crawler import CrawlerProcess
import re
import unicodedata
from string import punctuation
import os


##############################
#getting files_name in a list
def get_files_list(path):
    l = os.listdir(path)
    return [i for i in l if i.startswith("00")]

#word tokenize
def word_tokenizing(text):
    return word_tokenize(text)

#sentences tokenize
def sentences_tokenizing(text):
    return sent_tokenize(text)

#get excel_file column of filenames
def get_column(excel_file):
    df = pd.read_excel("./" + excel_file)
    return df.SECFNAME

#get urls  to scrape
def get_urls():
    l = []
    string = "https://www.sec.gov/Archives/"
    for i in get_column("cik_list.xlsx"):
        l.append(string + i)
    return l

#reading file using path +filename
def get_file(path_filename):
    with open(path_filename, 'r', encoding='utf-8') as f:
        return f.read()

#get list of lines in file given a path +filename
def get_file_list(path_filename):
    with open(path_filename, 'r') as f:
        return f.readlines()

# remove pounctuation
def strip_punctuation(text):
    return ''.join(c for c in text if c not in punctuation)

#remove html tags
def remove_tags1(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

#remove new line
def remove_newline(text):
    return re.sub('\n', '', text)

#remove tabulation
def remove_tab(text):
    return re.sub('\t', '', text)

# remove xa and so on
def remove_xa(text):
    new_str = unicodedata.normalize("NFKD", text)
    return new_str

#remove strange word
def remove_strange(text):
    text = text.replace("&#160", "")
    text = text.replace("Â", "")
    return text

#more cleaning to remove non-english words the longest word in english is 30 letters
def more_cleaning(text):
    nltk.download('words')
    words = set(nltk.corpus.words.words())
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if
                    w.lower() in words  or not w.isnumeric() )
    text = " ".join(w for w in text.split(" ") if not len(w) > 30)
    return text

#remove the spaces and lower the elements list
def stripandlower_list(list):
    return [w.strip().lower() for w in list]

#filter by l not in l1
def filter(l, l1):
    return [word.lower() for word in l if word.lower() not in l1 if word not in punctuation]

#get list using a query
def get_list_P_N(query, l):
    df = pd.read_csv("dictionary.csv")
    l1 = list(df.query(query).Word)
    return filter(l1, l)

#calculate score  positive negative  constraining uncertainty
def P_N_score1(filtered, l):
    return len([word for word in filtered if word in l])

#polarity score
def polarity_score1(p_score, n_score):
    return (p_score - n_score) / ((p_score + n_score) + 0.000001)

#subjectivity score
def subjectivity_score1(p_score, n_score, filtered):
    return (p_score + n_score) / ((len(filtered)) + 0.000001)

#classify based on polarity score
def sentiment_score_categorization1(polarity_score):
    if polarity_score < -0.5:
        return "Most Negative"
    elif 0 > polarity_score >= -0.5:
        return "Negative"
    elif polarity_score == 0:
        return "Neutral"
    elif 0 < polarity_score <= 0.5:
        return "Positive"
    elif polarity_score > 0.5:
        return "Very Positive"
    else:
        return "code is  not ok"

#ratio
def ratio1(l, l1):
    if len(l1)!=0:
        return len(l) / len(l1)
    else:
        return 0

#count syllable
def count_syllable(word):
    c = 0
    vowels = "AEIOUaeiou"

    if (not word.endswith("ed")) or (not word.endswith("es")):
        for i in word:
            if i in vowels:
                c += 1
    return c

#complex words list
def complex_words1(filtered):
    return [word for word in filtered if count_syllable(word) > 2]

#calculate fog_index
def fog_index1(average_sentence_length, percentage_complex_words):
    return 0.4 * (average_sentence_length + percentage_complex_words)
#proportion
def proportion1(score,word_count):
    if word_count==0:
        return 0

    return score/word_count
#scraping class
class Reports(scrapy.Spider):
    name = 'Reports'

    def start_requests(self):
        for i in get_urls():
            yield scrapy.Request(i, self.parse)

    def parse(self, response):
        root = lxml.html.fromstring(response.body)
        lxml.etree.strip_elements(root, lxml.etree.Comment, "script", "head")
        text = lxml.html.tostring(root, method="text", encoding="utf-8")
    #path where you want to store the files
        path = "C:/Users/hatem/Downloads/Data Science-20211015T140504Z-001/Data Science/files/"
        filename = path + response.url.split('/')[-1] + '.txt'
        with open(filename, 'wb') as f:
            f.write(text)


'''process = CrawlerProcess()
process.crawl(Reports)
process.start()'''

#################
#variables
path = "C:/Users/hatem/Downloads/Data Science-20211015T140504Z-001/Data Science/files/"
l=get_files_list(path)
stop_words = get_file_list(path + "StopWords_Generic.txt")
stop_words = stripandlower_list(stop_words)
l_negative = get_list_P_N("Negative>0", stop_words)
l_positive = get_list_P_N("Positive>0", stop_words)
df1=pd.read_excel("constraining_dictionary.xlsx")
df2=pd.read_excel("uncertainty_dictionary.xlsx")
constraining_list=list(df1.Word)
uncertainty_list=list(df2.Word)
constraining_list=filter(constraining_list,stop_words)
uncertainty_list=filter(uncertainty_list,stop_words)
j=0
s=0
#loop through all files
data = {"file":[],'positive_score': [], 'negative_score': [], 'polarity_score': [], 'subjectivity_score': [], "sentiment_score_categorization": [], 'average_sentence_length': [], 'complex_words': [],"word_count":[],"percentage_complex_words":[],"fog_index":[],"constraining_score":[],"uncertainty_score":[],"positive_score_propotion":[],"negative_score_propotion":[],"constraining_score_propotion":[],"uncertainty_score_propotion":[]}
for i in l:
    text = get_file(path + i)
    data["file"].append(i)
    text1 = str(text)
    text = remove_tags1(text)
    text = remove_xa(text)
    text = remove_newline(text)
    text = remove_tab(text)
    text = remove_strange(text)
    text = strip_punctuation(text)
    text = more_cleaning(text)
    tokens = word_tokenizing(text)
    filtered = filter(tokens, stop_words)
    positive_score=P_N_score1(filtered, l_positive)#1
    data["positive_score"].append(positive_score)
    negative_score=P_N_score1(filtered,l_negative)#2
    negative_score=-negative_score
    data["negative_score"].append(negative_score)
    polarity_score=polarity_score1(positive_score,negative_score)#3
    data["polarity_score"].append(polarity_score)
    subjectivity_score=subjectivity_score1(positive_score,negative_score,filtered)#4
    data["subjectivity_score"].append(subjectivity_score)
    s_s_c=sentiment_score_categorization1(polarity_score)#5
    data["sentiment_score_categorization"].append(s_s_c)
    text1=remove_tags1(text1)
    text1=remove_xa(text1)
    text1=remove_newline(text1)
    text1=remove_tab(text1)
    text1=remove_strange(text1)
    sentences=sentences_tokenizing(text1)
    average_sentence_length=ratio1(filtered,sentences)#6
    data['average_sentence_length'].append(average_sentence_length)
    complex_words=complex_words1(filtered)
    data["complex_words"].append(len(complex_words))
    percentage_complex_words=ratio1(complex_words,filtered)#7
    data["percentage_complex_words"].append(percentage_complex_words)
    fog_index=fog_index1(average_sentence_length,percentage_complex_words)#8
    data["fog_index"].append(fog_index)

    constraining_score=P_N_score1(filtered,constraining_list)#9
    data['constraining_score'].append(constraining_score)
    uncertainty_score=P_N_score1(filtered,uncertainty_list)#10
    uncertainty_score=-uncertainty_score
    data["uncertainty_score"].append(uncertainty_score)

    positive_score_propotion=proportion1(positive_score,len(filtered))#11
    data['positive_score_propotion'].append(positive_score_propotion)

    negative_score_propotion=proportion1(negative_score,len(filtered))#12
    data['negative_score_propotion'].append(negative_score_propotion)
    constraining_score_propotion=proportion1(constraining_score,len(filtered))#13
    data["constraining_score_propotion"].append(constraining_score_propotion)
    uncertainty_score_propotion=proportion1(uncertainty_score,len(filtered))#14
    data["uncertainty_score_propotion"].append(uncertainty_score_propotion)
    word_count=len(filtered)#15
    data["word_count"].append(word_count)
    s+=constraining_score
    j+=1

m=s/j
dataframe=pd.DataFrame.from_dict(data)
dataframe["constraining_words_whole_report"]=m
dataframe.to_csv("output data.csv",index=False)