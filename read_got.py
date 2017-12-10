# Reads the GOT dataset and returns it as a list of lists

import numpy as np
import os
from data_provider import *

class ReadGOT(DataProvider):
	def __init__(self, data_dir, rare_threshold=0, length_threshold=0):
		"""
			__init_(ReadGOT, str, int, int) -> None
			data_dir: Path to the directory containing data.txt
			rare_threshold: Only words which appear >= rare_threshold times will be retained, others will be
							replaced by the '__rare__' token
			length_threshold: Ignores sentences which contain fewer than length_threshold words
		"""
		DataProvider.__init__(self)
		self.data_dir = data_dir
		self.word_to_id = dict()
		self.id_to_word = dict()
		self.sentences = []
		self.num_sentences = 0
		self.num_words = 0
		self.word_count = dict()
		self.rare_threshold = rare_threshold
		self.length_threshold = length_threshold
		self.current_idx = 0
		self.read_data()
		
		
		
	def read_data(self):
		"""
			read_data(ReadGOT) -> None
		"""
		# Gather statistics
		self.gather_statistics()
		
		# Read the data
		f = open(os.path.join(self.data_dir, 'data.txt'))
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
		f.close()
		
		
		
	def gather_statistics(self):
		"""
			gather_statistics(ReadGOT) -> None
			Gathers the statistics related to the dataset
		"""
		f = open(os.path.join(self.data_dir, 'data.txt'))
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
		f.close()
		
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
			get_id(ReadGOT, str) -> int
			Returns the id corresponding to the given word
		"""
		return self.word_to_id[word]
		
	
	def get_word(self, idx):
		"""
			get_word(ReadGOT, int) -> str
			Returns the word corresponding to the given id
		"""
		return self.id_to_word[idx]
		
	
	
	def get_sentence(self, idx):
		"""
			get_sentence(ReadGOT, int) -> list
			Returns the sentence at the given index
		"""
		return self.sentences[idx][:]
		
	
	
	def get_next_sentence(self):
		"""
			get_next_sentence(ReadGOT) -> list
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
			get_word_count(ReadGOT, str, int) -> int
			Returns the number of time the given word appears in the training corpus
		"""
		if word != None:
			return self.word_count[word]
		elif idx != None:
			return self.word_count[self.id_to_word[idx]]
		else:
			return -1
		

if __name__ == '__main__':
	read_got = ReadGOT('./Data', rare_threshold=10, length_threshold=3)
	num_sentences = read_got.get_num_sentences()
	num_words = read_got.get_num_words()
	
	print num_sentences
	print num_words
	
	import random
	idx = random.choice(range(num_sentences))
	
	sentence = read_got.get_sentence(idx)
	print sentence
	for idx in sentence:
		print (read_got.get_word(idx), read_got.get_word_count(word=read_got.get_word(idx)), \
																		read_got.get_word_count(idx = idx))
	
	for i in range(5):
		sentence = read_got.get_next_sentence()
		s = [read_got.get_word(idx) for idx in sentence]
		print s
	
