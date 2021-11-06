import os
import numpy as np
import torch
import pandas as pd
import re
import unicodedata
from string import punctuation
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
nltk.download('words')
words = set(nltk.corpus.words.words())
l = [i for i in os.listdir("./") if i.startswith("00")]
text_list = []
for i in range(len(l)):
    with open(l[i], "r", encoding='utf-8') as f:
        text = f.read()
        text_list.append(text)


TAG_RE = re.compile(r'<[^>]+>')


def strip_punctuation(text):
    return ''.join(c for c in text if c not in punctuation)


def remove_tags1(text):
    return TAG_RE.sub('', text)


def remove_newline(text):
    return re.sub('\n', '', text)


def remove_tab(text):
    return re.sub('\t', '', text)


def remove_xa(text):
    new_str = unicodedata.normalize("NFKD", text)
    return new_str


def remove_strange(text):
    text = text.replace("&#160", "")
    text = text.replace("Â", "")
    return text





def join(text):
    sent = " ".join(
        w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isnumeric() or not w.isalnum())
    sent = " ".join(w for w in sent.split(" ") if not len(w) > 30)
    return sent


text_list_cleaned = []
for text in text_list:
    text = remove_tags1(text)
    text = remove_xa(text)
    text = remove_newline(text)
    text = remove_tab(text)
    text = remove_strange(text)
    text = join(text)
    text_list_cleaned.append(text)
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

summary_list = []
for i in text_list_cleaned:
    parser = PlaintextParser.from_string(i, Tokenizer('english'))
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document, 2)
    ch = ""
    # Printing the summary
    for sentence in lsa_summary:
        ch += str(sentence)
    summary_list.append(ch)


path = "C:/Users/hatem/Downloads/finBERT"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)


def predict(text):
    MAX_LEN = 160
    class_names = ['negative', 'neutral', 'positive']

    encoded_new = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LEN,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        # Return pytorch tensors.
        return_tensors='pt')

    # Add the encoded sentence to the list.
    input_idst = (encoded_new['input_ids'])
    attention_maskst = (encoded_new['attention_mask'])

    # Convert the lists into tensors.
    input_idst = torch.cat([input_idst], dim=0)
    attention_maskst = torch.cat([attention_maskst], dim=0)

    new_test_output = model(input_idst, token_type_ids=None,
                            attention_mask=attention_maskst)

    logits = new_test_output[0]
    predicted = logits.detach().numpy()

    # Store predictions
    flat_predictions = np.concatenate(predicted, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    new_predictions = np.argmax(flat_predictions).flatten()

    return class_names[new_predictions[0]]


predictions = []

for summary in summary_list:
    predictions.append(predict(summary))

data = {"file": l, "cleaned": text_list_cleaned, "summary": summary_list, "prediction": predictions}
dataframe = pd.DataFrame.from_dict(data)
dataframe.to_csv("output_data_finbert.csv", index=False)
