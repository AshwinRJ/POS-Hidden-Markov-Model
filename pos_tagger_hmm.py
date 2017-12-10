# Implements part of speech tagging using fully unsupervised HMM
from read_got import *
import random
import numpy as np
import os

class POSTagger(object):
	
	def __init__(self, data, resume=True):
		"""
			__init__(POSTagger, DataProvider, bool) -> None
			data: An instantiation of a subclass of DataProvider
			resume: Loads the last saved parameters if available
		"""
		self.data = data
		self.parameters_dir = './params/'
		self.num_labels = 20	# Number of POS tags
		
		# Initialize the label sequence randomly
		self.current_labels = []	# Label sequence for each example in training set
		self.init_labels()
		
		# Initialize the parameter matrices (or load them if necessary)
		if (not resume) or (not self.load_params()):
			self.e_prob = np.random.uniform(size=(self.num_labels, self.data.get_num_words()))
			self.e_prob = self.e_prob / np.reshape(np.sum(self.e_prob, axis=1), (self.num_labels, 1))	# Emission probabilities
			self.t_prob = np.random.uniform(size=(self.num_labels, self.num_labels))
			self.t_prob = self.t_prob / np.reshape(np.sum(self.t_prob, axis=1), (self.num_labels, 1))	# Transition probabilities
			self.p_prob = np.random.uniform(size=(self.num_labels))
			self.p_prob = self.p_prob / np.sum(self.p_prob)	# Prior probability for y1
		
		# Initialize alphas and betas
		self.alphas = []
		self.betas = []
		self.init_alpha_beta()
		
		# Initialize gammas and etas
		self.gammas = []
		self.etas = []
		self.init_gamma_eta()
		
	
	
	def train(self):
		"""
			train(POSTagger) -> float
			Trains the HMM
		"""
		print "\tForward Step"
		self.forward()	# Calculate alphas
		print "\tBackward Step"
		self.backward()	# Calculate betas
		print "\tCalculating P(x)"
		px = self.calculate_px()	# Calculate marginal prob
		print "\tCalculating Gammas"
		self.calculate_gammas(px)	# Calculate gammas
		print "\tCalculating Etas"
		self.calculate_etas(px)	# Calculate etas
		print "\tUpdating p_prob"
		self.update_p_prob()	# Update p_prob
		print "\tUpdating e_prob"
		self.update_e_prob()	# Update e_prob
		print "\tUpdating t_prob"
		self.update_t_prob()	# Update t_prob
		
		print "\tSaving Parameters"
		self.save_params()
		
		return np.sum(np.log(px + 1e-9))	# Return log p(x)
	
	
	
	def save_params(self):
		"""
			save_params(POSTagger) -> None
			Saves the parameters of the model
		"""
		f = open(os.path.join(self.parameters_dir, 'p_prob'), 'w')
		np.save(f, self.p_prob)
		f.close()
		f = open(os.path.join(self.parameters_dir, 'e_prob'), 'w')
		np.save(f, self.e_prob)
		f.close()
		f = open(os.path.join(self.parameters_dir, 't_prob'), 'w')
		np.save(f, self.t_prob)
		f.close()
	
	
	def load_params(self):
		"""
			load_params(POSTagger) -> bool
			Loads the parameters of the model if a saved file is found
			Returns:
				A boolean indicating if the loading was successful
		"""
		path_p = os.path.join(self.parameters_dir, 'p_prob')
		path_e = os.path.join(self.parameters_dir, 'e_prob')
		path_t = os.path.join(self.parameters_dir, 't_prob')
		if os.path.exists(path_p) and os.path.exists(path_e) and os.path.exists(path_t):
			self.p_prob = np.load(path_p)
			self.e_prob = np.load(path_e)
			self.t_prob = np.load(path_t)
			return True
		return False
	
	
	
	def update_t_prob(self):
		"""
			update_t_prob(POSTagger) -> None
			Updates the t_prob matrix
		"""
		sum_examples = np.zeros((self.num_labels, self.num_labels))
		for i in range(self.data.get_num_sentences()):
			sum_examples += np.sum(self.etas[i], axis=0)
		self.t_prob = sum_examples / np.reshape(np.sum(sum_examples, axis=1), (-1, 1))
	
	
	def update_e_prob(self):
		"""
			update_e_prob(POSTagger) -> None
			Updates the e_prob matrix
		"""
		sum_examples = np.zeros(np.shape(self.e_prob))
		for i in range(self.data.get_num_sentences()):
			sentence = self.data.get_sentence(i) 
			t_i = len(sentence)
			
			# Find which word is present in this sentence
			temp = np.zeros((t_i, np.shape(self.e_prob)[1]))	# t_i x vocab_size
			count = 0
			for word in sentence:
				temp[count, word] = 1
				count += 1
				
			# Calculate the probability
			gamma = self.gammas[i]	# t_i x num_labels	
			sum_examples += np.matmul(np.transpose(gamma), temp)	
			
		self.e_prob = sum_examples / np.reshape(np.sum(sum_examples, axis=1), (-1, 1))
		
		
	def update_p_prob(self):
		"""
			update_p_prob(POSTagger) -> None
			Updates the p_prob vector
		"""
		sum_examples = np.zeros((self.num_labels))
		for i in range(self.data.get_num_sentences()):
			sum_examples += self.gammas[i][0, :]		
		self.p_prob = sum_examples / np.sum(sum_examples)
		
		
	def calculate_gammas(self, px):
		"""
			calculate_gammas(POSTagger, ndarray) -> None
			Calculates gamma(z^(i) jk = 1) for all i, j, k
			px: num_examples x 1, p(x^(i)) for all i
		"""
		for i in range(self.data.get_num_sentences()):
			alpha = self.alphas[i]	# t_i x num_labels
			beta = self.betas[i]	# t_i-1 x num_labels
			self.gammas[i] = (alpha * \
				np.concatenate([beta, np.ones((1, self.num_labels))], axis=0)) / (px[i]+1e-9)

	
	def calculate_etas(self, px):
		"""
			calculate_etas(POSTagger, ndarray) -> None
			Calculates the eta(z^(i)km, z^(i)k-1,l) for all i, k, l, m
			px: num_examples x 1, p(x^(i)) for all i
		"""
		for i in range(self.data.get_num_sentences()):
			sentence = self.data.get_sentence(i)
			alpha = self.alphas[i]	# t_i x num_labels
			beta = self.betas[i]	# t_i-1 x num_labels
			t_i = np.shape(self.alphas[i])[0]
			temp = np.zeros((t_i-1, self.num_labels, self.num_labels))
			for k in range(2, t_i):
				temp[k-2, :, :] = np.matmul(\
					np.reshape(alpha[k-2, :], (self.num_labels, 1)), \
					np.reshape(beta[k-1, :], (1, self.num_labels))) * \
					(self.t_prob * np.reshape(self.e_prob[:, sentence[k-1]], (1, self.num_labels)))
			temp[t_i-2, :, :] = np.reshape(alpha[t_i-2, :], (self.num_labels, 1)) * \
					(self.t_prob * np.reshape(self.e_prob[:, sentence[t_i-1]], (1, self.num_labels)))
			self.etas[i] = temp / (px[i]+1e-9)
	
	
			
	def calculate_px(self):
		"""
			calculate_px(self) -> ndarray
			Calculates the probability of x^(i) for all examples i
			Returns:
				px: num_examples x 1 numpy array
		"""
		px = np.zeros((self.data.get_num_sentences(), 1))
		for i in range(self.data.get_num_sentences()):
			px[i, 0] = np.sum(self.alphas[i][-1, :])
		return px
		
		
	def forward(self):
		"""
			forward(POSTagger) -> None
			Implements the forward part of forward-backward algorithm
		"""		
		for i in range(self.data.get_num_sentences()):
			sentence = self.data.get_sentence(i)
			t_i = len(sentence)
			
			# Calculate alpha[0]
			self.alphas[i][0, :] = np.reshape(self.e_prob[:, sentence[0]], (1, self.num_labels)) * \
								np.reshape(self.p_prob, (1, self.num_labels))
			
			# Calculate other values of alpha recursively
			for j in range(1, t_i):
				self.alphas[i][j, :] = np.matmul(\
						np.reshape(self.alphas[i][j-1, :], (1, self.num_labels)), \
						self.t_prob) * \
						np.reshape(self.e_prob[:, sentence[j]], (1, self.num_labels))
						
						
	
	def backward(self):
		"""
			backward(POSTagger) -> None
			Implements the backward part of forward-backward algorithm
		"""		
		for i in range(self.data.get_num_sentences()):
			sentence = self.data.get_sentence(i)
			t_i = len(sentence)
			
			# Calculate betas[T-1]
			self.betas[i][t_i-2, :] = np.matmul(\
						np.reshape(self.e_prob[:, sentence[t_i-1]], (1, self.num_labels)), \
						np.transpose(self.t_prob))
			
			# Calculate other values of alpha recursively
			for j in range(t_i-3, -1, -1):
				self.betas[i][j, :] = np.matmul(\
						np.reshape(self.betas[i][j+1, :], (1, self.num_labels)) * \
						np.reshape(self.e_prob[:, sentence[j+1]], (1, self.num_labels)), \
						np.transpose(self.t_prob)) 
	
		
		
	def init_labels(self):
		"""
			init_labels(POSTagger) -> None
			Initializes the labels for all examples randomly
		"""
		for i in range(self.data.get_num_sentences()):
			sentence = self.data.get_next_sentence()
			tags = []
			for word in sentence:
				tags.append(random.choice(range(self.num_labels)))
			self.current_labels.append(tags)
			
	
	def init_alpha_beta(self):
		"""
			init_alpha_beta(POSTagger) -> None
			Initializes the alpha and beta for all examples
		"""
		for i in range(self.data.get_num_sentences()):
			sentence = self.data.get_sentence(i)
			t_i = len(sentence)
			self.alphas.append(np.zeros((t_i, self.num_labels)))
			self.betas.append(np.zeros((t_i-1, self.num_labels)))
			
	
	
	def init_gamma_eta(self):
		"""
			init_gamma_eta(POSTagger) -> None
			Initializes the gamma and eta for all examples
		"""
		for i in range(self.data.get_num_sentences()):
			sentence = self.data.get_sentence(i)
			t_i = len(sentence)
			self.gammas.append(np.zeros((t_i, self.num_labels)))
			self.etas.append(np.zeros((t_i-1, self.num_labels, self.num_labels)))
			
			
			
	def test(self, sentences):
		"""
			test(POSTagger, list of str) -> list of ndarray
			Calculates the gamma values provided sentences
			Returns:
				List of gamma values for each sentence
		"""
		# Get the sentences in the right format
		test_sentences = []
		for sentence in sentences:
			words = sentence.split(' ')
			curr_sentence = []
			for word in words:
				word = word.strip()
				curr_sentence.append(self.data.get_id(word))
			test_sentences.append(curr_sentence)
			
			
		# Calculate gammas
		alphas = self.forward_test(test_sentences)	# Calculate alphas
		betas = self.backward_test(test_sentences)	# Calculate betas
		px = self.calculate_px_test(test_sentences, alphas)	# Calculate marginal prob
		gammas = self.calculate_gammas_test(px, sentences, alphas, betas)	# Calculate gammas	
		
		return gammas
		
	
	
	def calculate_gammas_test(self, px, sentences, alphas, betas):
		"""
			calculate_gammas_test(POSTagger, ndarray, list, ndarray, ndarray) -> list of ndarray
			Calculates gamma(z^(i) jk = 1) for all i, j, k
			px: num_examples x 1, p(x^(i)) for all i
			Returns:
				gamma: list of t_i x num_labels numpy arrays
		"""
		gammas = []
		n = len(sentences)
		for i in range(n):
			alpha = alphas[i]	# t_i x num_labels
			beta = betas[i]	# t_i-1 x num_labels
			gammas.append((alpha * \
				np.concatenate([beta, np.ones((1, self.num_labels))], axis=0)) / (px[i]+1e-9))
		return gammas


	def calculate_px_test(self, sentences, alphas):
		"""
			calculate_px_test(POSTagger, list, ndarray) -> ndarray
			Calculates the probability of x^(i) for all examples i
			Returns:
				px: num_examples x 1 numpy array
		"""
		n = len(sentences)
		px = np.zeros((n, 1))
		for i in range(n):
			px[i, 0] = np.sum(alphas[i][-1, :])
		return px
	
	
	def forward_test(self, sentences):
		"""
			forward_test(POSTagger, list) -> list of ndarray
			Implements the forward part of forward-backward algorithm
		"""	
		n = len(sentences)
		alphas = []
		for i in range(n):
			sentence = sentences[i]
			t_i = len(sentence)
		
			# Initialize alpha
			alpha = np.zeros((t_i, self.num_labels))
			
			# Calculate alpha[0]
			alpha[0, :] = np.reshape(self.e_prob[:, sentence[0]], (1, self.num_labels)) * \
								np.reshape(self.p_prob, (1, self.num_labels))
		
			# Calculate other values of alpha recursively
			for j in range(1, t_i):
				alpha[j, :] = np.matmul(\
						np.reshape(alpha[j-1, :], (1, self.num_labels)), \
						self.t_prob) * \
						np.reshape(self.e_prob[:, sentence[j]], (1, self.num_labels))
			
			alphas.append(alpha)
		return alphas
					
					

	def backward_test(self, sentences):
		"""
			backward_test(POSTagger, list) -> list of ndarray
			Implements the backward part of forward-backward algorithm
		"""		
		n = len(sentences)
		betas = []
		for i in range(n):
			sentence = sentences[i]
			t_i = len(sentence)
			
			# Initialize beta
			beta = np.zeros((t_i-1, self.num_labels))
		
			# Calculate betas[T-1]
			beta[t_i-2, :] = np.matmul(\
						np.reshape(self.e_prob[:, sentence[t_i-1]], (1, self.num_labels)), \
						np.transpose(self.t_prob))
		
			# Calculate other values of alpha recursively
			for j in range(t_i-3, -1, -1):
				beta[j, :] = np.matmul(\
						np.reshape(beta[j+1, :], (1, self.num_labels)) * \
						np.reshape(self.e_prob[:, sentence[j+1]], (1, self.num_labels)), \
						np.transpose(self.t_prob))
			
			betas.append(beta)			
		return betas
	


if __name__ == '__main__':
	from dummy_data import *
	#data = DummyData()
	#pos_tagger = POSTagger(data, False)
	data = ReadGOT('./Data', 10, 3)
	pos_tagger = POSTagger(data)
	
	# Test
	"""
	sentences = ['it was a good day for khaleesi',
	            'someone wanted to kill ned stark',
	            'arya stark became blind',
	            'jon snow was king of the north']
	gammas = pos_tagger.test(sentences)
	for gamma in gammas:
		print np.argmax(gamma, axis=1)
	"""
	
	# Train
	#"""
	for i in range(100):
		print "Training Epoch:", i+1
		cost = pos_tagger.train()
		print cost
	#"""
