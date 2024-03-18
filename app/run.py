import json
import os
from dotenv import load_dotenv
from preprocessing import Preprocessor
from clustering_model import ClusterModel
from llm import LLMTextOrderer
import pandas as pd
import numpy as np
import itertools
from sentence_transformers import SentenceTransformer
from embedding_model import finetune_sentence_transformer
import warnings

warnings.filterwarnings("ignore")


def validate_configuration():
	try:
		assert os.getenv('DIALOGUES_FILE').split('.')[0] in os.getenv('COLUMNS').keys()
	except AssertionError:
		print('Dialogues file name not in sync with df dialogue column name')
	try:
		assert os.getenv('SUMMARY_PIECES_FILE').split('.')[0] in os.getenv('COLUMNS').keys()
	except AssertionError:
		print('Summary pieces file name not in sync with df summary pieces column name')


def cosine_similarity(a, b):
	return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run():
	validate_configuration()

	if os.getenv('CREATE_EMBEDDINGS'):
		embedding_model = finetune_sentence_transformer()
	else:
		embedding_model = SentenceTransformer(os.getenv('EMBEDDING_MODEL'))

	dialogues = Preprocessor(file=os.getenv('DIALOGUES_FILE'),
							 ner=True,
							 model=embedding_model)
	dialogues.process_text()
	pieces = (Preprocessor(file=os.getenv('SUMMARY_PIECES_FILE'),
						  ner=True,
						  model=embedding_model))
	pieces.process_text()

	ClusterModel()
	pieces_df = cluster_pieces(num_clusters=len(dialogues.df),
							   df=pieces.df)

	results = []
	for cluster in pieces_df.cluster_assignment.values:
		clustered_sentences = pieces_df[pieces_df['cluster_assignment'] == cluster]['pieces'].values.tolist()
		encoded_cluster = embedding_model.encode('.'.join(clustered_sentences))

		dialogues.df["similarity"] = dialogues.df['embedding'].apply(lambda x: cosine_similarity(x, encoded_cluster))
		top_dialogue_match_df = dialogues.df.sort_values("similarity", ascending=False).head(1)
		dialogue_id = top_dialogue_match_df['id'].values

		llm = LLMTextOrderer()
		ordered_summary = llm.order_sentences(sentences=clustered_sentences,
											  text=top_dialogue_match_df['dialogue'].values)
		results.append([(dialogue_id, sentence, index) for index, sentence in enumerate(ordered_summary.split('. '))])

	final = list(itertools.chain.from_iterable(results))
	final_df = pd.DataFrame(final, columns=['dialogue_id', 'summary_piece', 'position_index'])
	final_df.to_csv(r".\results.csv")


if __name__ == '__main__':
	load_dotenv(r"../.env")

	api_keys_path = os.path.join(os.path.expanduser("~"), os.getenv('TOKEN_PATH'))

	with open(api_keys_path, 'r') as file:
		api_key_dict = json.loads(file.read())

	os.environ["OPENAI_API_KEY"] = api_key_dict['openai']
	run()
