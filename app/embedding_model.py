from sentence_transformers import InputExample, losses, SentenceTransformer, models
from torch.utils.data import DataLoader
import os
from preprocessing import Preprocessor


class FinetunedSentenceTransformer:
    def __init__(self, base_model: str, train_data_loader, epochs: int= 10, percent_train: float= 0.80):
        self.base_model = base_model
        self.epochs = epochs
        self.percent_train = percent_train
        self.train_data_loader = train_data_loader

    def train_model(self):
        embedding_model = models.Transformer(self.base_model)

        pooling_model = models.Pooling(embedding_model.get_word_embedding_dimension())

        model = SentenceTransformer(modules=[embedding_model,
                                             pooling_model])

        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        warmup_steps = int(len(self.train_data_loader) * self.epochs * 0.1)

        model.fit(train_objectives=[(self.train_data_loader, train_loss)],
                  epochs=self.epochs,
                  warmup_steps=warmup_steps)

        return model


def load_training_data():
    ref_dialogues = Preprocessor(file=os.getenv('REF_DIALOGUES_FILE'),
                                 ner=False,
                                 embed=False,
                                 model=None)
    ref_dialogues.process_text()

    ref_pieces = Preprocessor(file=os.getenv('REF_SUMMARY_PIECES_FILE'),
                                 ner=False,
                                 embed=False,
                                 model=None)
    ref_pieces.process_text()

    training_data = []
    for _, row in ref_pieces.df.iterrows():
        training_data.append(InputExample(
            texts=[row['cleaned_text'], ref_dialogues.df[ref_dialogues.df['id'] == row['dialog_id']]['cleaned_text'].values]))

    return DataLoader(training_data, shuffle=True, batch_size=16)


def finetune_sentence_transformer():
    train_loader = load_training_data()

    finetuner = FinetunedSentenceTransformer(base_model=os.getenv("EMBEDDING_MODEL"),
                                             train_data_loader=train_loader,
                                             )
    model = finetuner.train_model()

    return model