import torch
from torch import nn


class NewsInterpreter(nn.Module):
    def __init__(self, nli_model, nli_tokenizer, roberta_model, roberta_tokenizer, device):
        super(NewsInterpreter, self).__init__()
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.roberta_model = roberta_model
        self.roberta_tokenizer = roberta_tokenizer
        self.device = device

        self.nli_model.to(self.device)
        self.nli_model.eval()
        self.roberta_model.to(self.device)
        self.roberta_model.eval()

    def preprocess(self, text: str):
        import re

        # remove mentions
        text = re.sub(r'@\w+', "", text)
        # remove links
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # remove \n
        text = re.sub(r'\n', '', text)
        # remove emojis
        emoj = re.compile("["
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                      "]+", re.UNICODE)
        return re.sub(emoj, '', text)

    def forward(self, news_texts: list):
        # Extract text from each tuple (text, date, type)
        texts = [self.preprocess(text) for text in news_texts[0]]


        # NLI embeddings
        hypothesis_positive = "The Bitcoin Price is likely to continue rising."
        nli_encoded_inputs = self.nli_tokenizer(
            texts,
            [hypothesis_positive] * len(texts),
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            nli_outputs = self.nli_model(**nli_encoded_inputs)
            nli_last_hidden_state = nli_outputs.last_hidden_state
            nli_pooled_embeddings = nli_last_hidden_state.mean(dim=1)

        # RoBERTa embeddings
        roberta_encoded_inputs = self.roberta_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            roberta_outputs = self.roberta_model(**roberta_encoded_inputs)
            roberta_last_hidden_state = roberta_outputs.last_hidden_state
            roberta_pooled_embeddings = roberta_last_hidden_state.mean(dim=1)

        # Combine the embeddings
        combined_embeddings = torch.cat((nli_pooled_embeddings, roberta_pooled_embeddings), dim=1)

        return combined_embeddings
