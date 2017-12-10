# Base class for providing data to HMM
class DataProvider(object):
	def __init__(self):
		"""
			__init_(DataProvider) -> None
		"""
		pass
			
	
	def get_id(self, word):
		"""
			get_id(DataProvider, str) -> int
			Returns the id corresponding to the given word
		"""
		pass
		
	
	def get_word(self, idx):
		"""
			get_word(DataProvider, int) -> str
			Returns the word corresponding to the given id
		"""
		pass
		
	
	
	def get_sentence(self, idx):
		"""
			get_sentence(DataProvider, int) -> list
			Returns the sentence at the given index
		"""
		pass
		
	
	
	def get_next_sentence(self):
		"""
			get_next_sentence(DataProvider) -> list
			Returns the next sentence from the list
		"""
		pass
	
	
	def get_num_sentences(self):
		pass
		
	def get_num_words(self):
		pass
		
	
	def get_word_count(self, word=None, idx=None):
		"""
			get_word_count(DataProvider, str, int) -> int
			Returns the number of time the given word appears in the training corpus
		"""
		pass
