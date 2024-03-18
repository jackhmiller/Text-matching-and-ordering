from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import load_train_data, get_max_sequence_length


BATCH_SIZE = 16
EPOCHS = 10
NUM_LAYERS = 1


class TextPairDataset(Dataset):
	def __init__(self, text_pairs, tokenizer, max_length):
		self.text_pairs = text_pairs
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __getitem__(self, idx):
		text_pair = self.text_pairs[idx]
		encoding = self.tokenizer(text_pair, padding='max_length', truncation=True, max_length=self.max_length,
								  return_tensors='pt')
		return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0)}


class SiameseNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super(SiameseNetwork, self).__init__()
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
		self.fc1 = nn.Linear(hidden_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, 1)

	def forward_one(self, token_ids, attention_mask):
		embedded = self.embedding(token_ids)
		embedded = embedded * attention_mask.unsqueeze(-1)
		_, (h, _) = self.lstm(embedded)

		return h.squeeze(0)

	def forward(self, data):
		input_ids_A = data['input_ids'][:, 0]
		attention_mask_A = data['attention_mask'][:, 0]

		input_ids_B = data['input_ids'][:, 1]
		attention_mask_B = data['attention_mask'][:, 1]

		output_A = self.forward_one(input_ids_A, attention_mask_A)
		output_B = self.forward_one(input_ids_B, attention_mask_B)

		# distance = torch.sqrt(torch.sum(torch.pow(output1 - output2, 2), 1))
		# out = F.relu(self.fc1(distance))
		# out = self.fc2(out)

		return output_A, output_B


def get_bert_embeddings(model, tokenizer):
	vocab_size = model.config.vocab_size
	embedding_dim = model.config.hidden_size

	embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

	for token_id in range(vocab_size):
		token = tokenizer.convert_ids_to_tokens(token_id)
		with torch.no_grad():
			token_embedding = model.get_input_embeddings()(torch.tensor(token_id).unsqueeze(0))
		embedding_layer.weight.data[token_id] = token_embedding.squeeze(0)

	return embedding_layer


def train_model(pairs: list[str], tokenizer, max_length: int):
	dataset = TextPairDataset(pairs, tokenizer, max_length)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	input_size = tokenizer.vocab_size
	hidden_size = max_length

	model = SiameseNetwork(input_size, hidden_size, NUM_LAYERS)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	loss_function = nn.CosineSimilarity()

	losses = []
	for epoch in range(EPOCHS):
		for i, data in enumerate(dataloader):
			print(i)
			h1, h2 = model(data)
			loss = loss_function(h1, h2)
			print(f"Iteration {i}: loss = {loss}")
			losses.append(loss)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	torch.save(model, 'model.pth')
	return model


def run_network():
	training_pairs = load_train_data()
	bert_model_name = 'bert-base-uncased'
	tokenizer = BertTokenizer.from_pretrained(bert_model_name)
	bert_model = BertModel.from_pretrained(bert_model_name)

	max_length = get_max_sequence_length(tokenizer=tokenizer,
										 data=training_pairs)

	train_model(training_pairs, tokenizer, max_length)

	#embeddings = get_bert_embeddings(bert_model, tokenizer)



if __name__ == '__main__':
	run_network()