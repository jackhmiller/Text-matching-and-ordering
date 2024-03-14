from sentence_transformers import InputExample, losses, SentenceTransformer, models
from torch.utils.data import DataLoader


class FinetunedSentenceTransformer:

train_examples = []
train_data = dataset['train']['set']
# For agility we only 1/2 of our available data
n_examples = dataset['train'].num_rows // 2

for i in range(n_examples):
  example = train_data[i]
  train_examples.append(InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

## Step 1: use an existing language model
word_embedding_model = models.Transformer('distilroberta-base')

## Step 2: use a pool function over the token embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

## Join steps 1 and 2 using the modules argument
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_loss = losses.MultipleNegativesRankingLoss(model=model)

num_epochs = 10

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps)