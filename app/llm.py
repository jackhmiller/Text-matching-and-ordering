import pandas as pd
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.prompts.chat import (
	ChatPromptTemplate,
	SystemMessagePromptTemplate,
	HumanMessagePromptTemplate,
)


class LLMTopicModel:
	def __init__(self):
		self.system_template: str = "You're an expert journalist. You're helping me write a compelling topic title for news articles."
		self.user_template: str = "Using the following pieces of text, write a topic title that summarizes them.\n\nTEXT:{texts}\n\nTOPIC TITLE:"
		self.prompt = ChatPromptTemplate(messages=[SystemMessagePromptTemplate.from_template(self.system_template),
												   HumanMessagePromptTemplate.from_template(self.user_template),
												   ],
										 input_variables=["articles"]
										 )
		self.model = ChatOpenAI(temperature=0,
								model_name="gpt-3.5-turbo")

	def model_topics(self, label_col: str, df: pd.DataFrame) -> pd.DataFrame:
		df['topic'] = None
		for c in df[label_col].unique():
			chain = LLMChain(llm=self.model,
							 prompt=self.prompt,
							 verbose=False)

			clustered_sentences = '.'.join(df[df[label_col] == c]['pieces'].values.tolist())

			result = chain.run(
				{
					"texts": clustered_sentences,
				}
			)
			df.loc[df.label_col == c, "topic_title"] = result

		return df


class LLMTextOrderer:
	def __init__(self):
		self.client = OpenAI()
		self.llm = "gpt-3.5-turbo"

	def order_sentences(self, sentences, text):
		messages = [
			{"role": "system",
			 "content": "You are a helpful assistant designed to combine individual sentences into a single piece of text that represents a summary of a dialogue."
			 },
			{"role": "user",
			 "content": f"Please order the following sentences correctly so that they reflect the underling dialogue. Here are the sentences as a list to be ordered: {sentences}, and here is the underlying text: {text}"
			 }
		]

		response = self.client.chat.completions.create(
			model=self.llm,
			messages=messages)

		generated_summary = response.choices[0].message.content
		return generated_summary
