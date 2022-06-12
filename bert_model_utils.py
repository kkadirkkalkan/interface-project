import numpy as np
import torch
import json
from transformers import RobertaModel, RobertaTokenizer
from torch import cuda
import transformers
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import torch
import pandas as pd
import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=False)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.softmax= nn.Softmax(dim=1)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    output = self.out(output)
    return self.softmax(output)

#issue_title = "the last run of the code version"
#issue_desc = "the last run trial of the code does not work in the local environment"

#concanatated = issue_title + " " + issue_desc

def run_bert_model(issue_text):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    alist = []
    alist.append(issue_text)
    other_tokenizer = transformers.BertTokenizer("./vocab_bert.txt")

    map_location=torch.device('cpu')

    

    loaded_model = torch.load('./bert_sentiment.bin', map_location)

    tokens_late_test = other_tokenizer.batch_encode_plus(
        alist,
        max_length = 200,
        pad_to_max_length=True,
        truncation=True
    )

    




    sample_test_seq = torch.tensor(tokens_late_test['input_ids'])
    sample_test_mask = torch.tensor(tokens_late_test['attention_mask'])

    # get predictions for test data
    with torch.no_grad():
        preds1 = loaded_model(sample_test_seq.to(device), sample_test_mask.to(device))
        preds1 = preds1.detach().cpu().numpy()


    preds1 = np.argmax(preds1, axis = 1)

    return preds1[0]

