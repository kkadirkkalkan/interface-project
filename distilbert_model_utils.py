import numpy as np
import torch
import json
from transformers import DistilBertTokenizer, DistilBertModel
from torch import cuda

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


#issue_title = "the last run of the code version"
#issue_desc = "the last run trial of the code does not work in the local environment"

#concanatated = issue_title + " " + issue_desc

def run_distilbert_model(issue_text):
    device = 'cuda' if cuda.is_available() else 'cpu'


    alist = []
    alist.append(issue_text)
    other_tokenizer = DistilBertTokenizer("./distilbet_vocab.txt")

    map_location=torch.device('cpu')

    loaded_model = torch.load('./pytorch_distilbert_sentiment_trans.bin', map_location)

    tokens_late_test = other_tokenizer.batch_encode_plus(
        alist,
        max_length = 256,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=True
    )

    test_late_seq = torch.tensor(tokens_late_test['input_ids'])
    test_late_mask = torch.tensor(tokens_late_test['attention_mask'])
    test_late_token_id= torch.tensor(tokens_late_test['token_type_ids'])

    # get predictions for test data
    with torch.no_grad():
        pred_late = loaded_model(test_late_seq .to(device), test_late_mask.to(device), test_late_token_id.to(device))
        pred_late = pred_late.detach().cpu().numpy()


    pred_late = np.argmax(pred_late, axis = 1)

    return pred_late[0]

#print(run_distilbert_model(concanatated))