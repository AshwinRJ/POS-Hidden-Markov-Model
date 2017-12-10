# Dummy Dataset

import numpy as np
from data_provider import *

class DummyData(DataProvider):
	def __init__(self):
		"""
			__init_(DummyData) -> None
		"""
		DataProvider.__init__(self)
		self.word_to_id = dict()
		self.id_to_word = dict()
		self.sentences = []
		self.num_sentences = 0
		self.num_words = 0
		self.word_count = dict()
		self.current_idx = 0
		self.length_threshold = 0
		self.rare_threshold = 0
		self.read_data()
		
		
		
	def read_data(self):
		"""
			read_data(DummyData) -> None
		"""
		# Gather statistics
		self.gather_statistics()
		
		# Read the data
		f = ['this is a dummy dataset', \
			'this dataset is dummy', \
			'any number of sentences can be inserted here', \
			'dummy dataset will be used for any number of purposes']
		for sentence in f:
			words = sentence.strip().split(" ")
			if len(words) >= self.length_threshold:
				curr_sentence = []
				for word in words:
					if len(word.strip()) > 0:
						if self.word_count[word] >= self.rare_threshold:
							curr_sentence.append(self.word_to_id[word])
						else:
							curr_sentence.append(self.word_to_id['__rare__'])
				self.sentences.append(curr_sentence)
		
		
		
	def gather_statistics(self):
		"""
			gather_statistics(DummyData) -> None
			Gathers the statistics related to the dataset
		"""
		f = ['this is a dummy dataset', \
			'this dataset is dummy', \
			'any number of sentences can be inserted here', \
			'dummy dataset will be used for any number of purposes']
		for sentence in f:
			words = sentence.strip().split(" ")
			if len(words) >= self.length_threshold:
				self.num_sentences += 1
				for word in words:
					if len(word.strip()) > 0:
						try:
							self.word_count[word] += 1
						except:
							self.word_count[word] = 1
		
		
		# Convert from word to ID and ID to word
		idx = 0
		rare_count = 0
		for word in self.word_count:
			if self.word_count[word] >= self.rare_threshold:
				self.word_to_id[word] = idx
				self.id_to_word[idx] = word
				idx += 1
			else:
				rare_count += 1	# Accumulate the rare word count
		
		# Add rare to the word lsit
		self.word_to_id['__rare__'] = idx
		self.id_to_word[idx] = '__rare__'
		
		# Calculate the number of words		
		self.num_words = len(self.word_to_id)
		
		# Add rare_count to word_count
		self.word_count['__rare__'] = rare_count
		
			
	
	def get_id(self, word):
		"""
			get_id(DummyData, str) -> int
			Returns the id corresponding to the given word
		"""
		return self.word_to_id[word]
		
	
	def get_word(self, idx):
		"""
			get_word(DummyData, int) -> str
			Returns the word corresponding to the given id
		"""
		return self.id_to_word[idx]
		
	
	
	def get_sentence(self, idx):
		"""
			get_sentence(DummyData, int) -> list
			Returns the sentence at the given index
		"""
		return self.sentences[idx][:]
		
	
	
	def get_next_sentence(self):
		"""
			get_next_sentence(DummyData) -> list
			Returns the next sentence from the list
		"""
		s = self.sentences[self.current_idx]
		self.current_idx = (self.current_idx + 1) % self.num_sentences
		return s[:]
	
	
	def get_num_sentences(self):
		return self.num_sentences
		
	def get_num_words(self):
		return self.num_words
		
	
	def get_word_count(self, word=None, idx=None):
		"""
			get_word_count(DummyData, str, int) -> int
			Returns the number of time the given word appears in the training corpus
		"""
		if word != None:
			return self.word_count[word]
		elif idx != None:
			return self.word_count[self.id_to_word[idx]]
		else:
			return -1
