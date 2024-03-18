import re
import string
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import spacy

NLP = spacy.load("en_core_web_sm")


class Preprocessor:
    def __init__(self, file: str, ner: bool, embed: bool, model):
        self.ner = ner
        self.df = pd.read_csv(os.path.join(os.getenv('DATA_PATH'), file))
        self.text_column = os.getenv('COLUMNS')[file.split('.')[0]]
        self.embed = embed
        self.embedding_model = model

        if embed:
            try:
                assert model is not None
            except AssertionError:
                raise "Please provide an embedding model"

    @staticmethod
    def clean_text(text) -> str:
        text = text.lower()
        text = text.strip()
        text = re.compile('<.*?>').sub('', text)
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub(r'\[[0-9]*\]',' ',text) #[0-9]
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        text = re.sub(r'\d',' ',text)
        text = re.sub(r'\s+',' ',text)
        text = ' '.join([token.text for token in NLP(text) if not token.is_stop and len(token) > 1])
        return text

    def calc_max_tokens(self) -> int:
        self.df['text_length'] = self.df['cleaned_text'].apply(len)
        df_sorted = self.df.sort_values(by="text_length").reset_index(drop=True)
        longest_text = df_sorted[-1:]['cleaned_text'].values[0]
        return len(self.embedding_model.tokenize(longest_text))

    def embed_text(self) -> list:
        texts = self.df['cleaned_text'].values
        if self.ner:
            for i in range(len(texts)):
                doc = NLP(texts[i])
                for ent in doc.ents:
                    texts[i] += f' {ent.text}'

        return [self.embedding_model.encode(text) for text in texts]

    def process_text(self) -> None:
        self.df['cleaned_text'] = self.df[self.text_column].apply(self.clean_text)
        if self.embed:
            self.df['embedding'] = self.embed_text()
