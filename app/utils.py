import pandas as pd
from preprocessing import Preprocessor
import os


def create_training_pairs(pieces: pd.DataFrame, dialogues: pd.DataFrame) -> list:
	merged_df = pd.merge(pieces, dialogues, left_on='dialog_id', right_on='id', how='left')

	return list(zip(list(merged_df['cleaned_pieces'].values), list(merged_df['cleaned_dialogue'].values)))


def load_train_data() -> list:
	train_dialogues = Preprocessor(file=os.getenv('REF_DIALOGUES_FILE'),
							 ner=True,
							 embed=False
							 )
	train_dialogues.process_text()
	train_dialogues.df.rename(columns={'cleaned_text':'cleaned_dialogue'},
							  inplace=True)

	train_pieces = Preprocessor(file=os.getenv('REF_SUMMARY_PIECES_FILE'),
								ner=True,
								embed=True
								)
	train_pieces.process_text()
	train_dialogues.df.rename(columns={'cleaned_text': 'cleaned_pieces'},
							  inplace=True)

	training_data = create_training_pairs(pieces=train_pieces.df,
										  dialogues=train_dialogues.df)

	return training_data


def get_max_sequence_length(tokenizer, data: list[str]) -> int:
	sequence_lengths = []
	for pair in data:
		for element in pair:
			tokens = tokenizer.tokenize(element)
			sequence_lengths.append(len(tokens))

	max_sequence_length = max(sequence_lengths)

	return max_sequence_length
