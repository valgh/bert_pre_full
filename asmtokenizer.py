#############################
#ASM Tokenizer class
#############################
import os
import tqdm
import multiprocessing as mp
from multiprocessing import Manager


class ASMTokenizer:
	""" Tokenizer for ASM language to give as input to BERT """
	def __init__(self):
		self.vocab = {}
		self.decode_vocab = {}
		self.default_file_name = 'vocab_asm.txt'
		self.min_frequency = 1
		self.max_tokens = 32000


	def get_vocab_length(self):
		return len(self.vocab)


	def get_special_tokens_mask(self, token_ids_0, token_ids_1 = None, already_has_special_tokens: bool = False):
		if already_has_special_tokens:
			if token_ids_1 is not None:
				raise ValueError(
					"You should not supply a second sequence if the provided sequence of "
					"ids is already formatted with special tokens for the model."
				)
			return list(map(lambda x: 1 if x in [self.vocab['[SEP]'], self.vocab['[CLS]']] else 0, token_ids_0))

		if token_ids_1 is not None:
			return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
		return [1] + ([0] * len(token_ids_0)) + [1]


	def mp_process_line(self, line, vocabulary):
		if line != '\n':
			line.strip()
			words = line.split() # split on space char
			for word in words:
				word = word.decode("utf-8")
				if word not in vocabulary:
					vocabulary[word] = 1
				else:
					vocabulary[word] += 1
		return


	def process_wrapper(self, path_to_vocab, chunkStart, chunkSize, vocabulary):
		with open(path_to_vocab, 'rb') as f:
			f.seek(chunkStart)
			lines = f.read(chunkSize).splitlines()
			for line in lines:
				self.mp_process_line(line, vocabulary)


	def chunkify(self, fname, size=8*8): #size=1024*1024
		fileEnd = os.path.getsize(fname)
		with open(fname, 'rb') as f:
			chunkEnd = f.tell()
			while True:
				chunkStart = chunkEnd
				f.seek(size,1)
				f.readline()
				chunkEnd = f.tell()
				yield chunkStart, chunkEnd - chunkStart
				if chunkEnd > fileEnd:
					break


	def mp_train(self, path_to_vocab, min_frequency, max_tokens, pools_default):
		self.min_frequency = min_frequency
		self.max_tokens = max_tokens
		manager = Manager()
		vocabulary = manager.dict()
		pool = mp.Pool(pools_default)
		for chunkStart, chunkSize in tqdm.tqdm(self.chunkify(path_to_vocab)):
			pool.apply_async(self.process_wrapper, (path_to_vocab, chunkStart, chunkSize, vocabulary,))
		pool.close()
		pool.join()
		print('Training process complete.\n')
		path_to_save = os.getcwd()
		self.mp_save(path_to_save, vocabulary)
		return


	def mp_save(self, path_to_save, vocabulary):
		assert isinstance(path_to_save, str)
		if not os.path.exists(path_to_save):
			print('ERROR: Invalid path input.')
			return None
		if not vocabulary:
			print('ERROR: you have not trained this tokenizer!')
			return
		print('Saving results, might take a while...')
		to_save_vocab = sorted(vocabulary.items(),
							key=lambda item: item[1],
							reverse=True)
		count = 5
		with tqdm.tqdm(self.max_tokens) as pbar:
			with open(path_to_save+'\\'+'mp_'+self.default_file_name, 'a') as f:
				f.write('[CLS]')
				f.write('\n')
				f.write('[SEP]')
				f.write('\n')
				f.write('[UNK]')
				f.write('\n')
				f.write('[PAD]')
				f.write('\n')
				f.write('[MASK]')
				f.write('\n')
				for v in to_save_vocab:
					if v[1] >= self.min_frequency:
						count+=1
						f.write(v[0])
						f.write('\n')
						pbar.update(1)
					if count > self.max_tokens:
						break
		print('Vocab file saved under your cwd as {}'.format(self.default_file_name))
		print('\nReload the tokenizer with load_pretrained method to use it.')
		return

	def process_line(self, line):
		words = line.split() # split on space char
		for word in words:
			if word not in self.vocab:
				self.vocab[word] = 1
			else:
				self.vocab[word] += 1
		return


	def train(self, path_to_file, min_frequency=1, max_tokens=32000, multiprocessing=False, pools_default=2):
		assert isinstance(path_to_file, str)
		if not os.path.exists(path_to_file):
			print('ERROR: Invalid path input.')
			return None
		if multiprocessing:
			print('Begin multiprocessing.\n')
			return self.mp_train(path_to_file, min_frequency, max_tokens, pools_default)
		else:
			self.min_frequency = min_frequency
			self.max_tokens = max_tokens
			with tqdm.tqdm(os.path.getsize(path_to_file)) as pbar:
				with open(path_to_file, 'r') as f:
					for line in f:
						if line != '\n':
							pbar.update(len(line))
							self.process_line(line.strip())
			print('Training process complete.\n')
			self.save(os.getcwd())
			return


	def save(self, path_to_save):
		assert isinstance(path_to_save, str)
		if not os.path.exists(path_to_save):
			print('ERROR: Invalid path input.')
			return None
		if not self.vocab:
			print('ERROR: you have not trained this tokenizer!')
			return
		print('Saving results, might take a while...')
		to_save_vocab = sorted(self.vocab.items(),
							key=lambda item: item[1],
							reverse=True)
		count = 5
		with tqdm.tqdm(self.max_tokens) as pbar:
			with open(path_to_save+'\\'+self.default_file_name, 'a') as f:
				f.write('[CLS]')
				f.write('\n')
				f.write('[SEP]')
				f.write('\n')
				f.write('[UNK]')
				f.write('\n')
				f.write('[PAD]')
				f.write('\n')
				f.write('[MASK]')
				f.write('\n')
				for v in to_save_vocab:
					if v[1] >= self.min_frequency:
						count+=1
						f.write(v[0])
						f.write('\n')
						pbar.update(1)
					if count > self.max_tokens:
						break
		print('Vocab file saved under your cwd as {}'.format(self.default_file_name))
		print('\nReload the tokenizer with load_pretrained method to use it.')
		return


	def load_pretrained(self, path_to_pretrained):
		assert isinstance(path_to_pretrained, str)
		if not os.path.exists(path_to_pretrained):
			print('ERROR: Invalid path input.')
			return None
		self.vocab = {}
		self.decode_vocab = {}
		index = 0
		print('Loading vocabulary, might take a while...\n')
		with tqdm.tqdm(os.path.getsize(path_to_pretrained)) as pbar:
			with open(path_to_pretrained, 'r') as f:
				for line in f:
					self.vocab[line.strip()] = index
					self.decode_vocab[index] = line.strip()
					index+=1
					pbar.update(len(line))
		print('Tokenizer loaded.')
		return


	def encode(self, line, max_length=300, truncate=True, padding=True):
		""" Can be used to only encode without masks/token_ids"""
		out = [self.vocab['[CLS]']]
		words = line.split()
		for word in words:
			if word in self.vocab:
				out.append(self.vocab[word])
			else:
				out.append(self.vocab['[UNK]'])
		if truncate and len(out) >= max_length-1:
			out = out[:max_length-1]mto
		if len(out) < max_length-1 and padding:
			while len(out) != max_length-1:
				out.append(self.vocab['[PAD]'])
		out.append(self.vocab['[SEP]'])
		return out
		

	def decode(self, seq):
		output = ''
		for s in seq:
			output += self.decode_vocab[s]+' '
		return output



	def __call__(self, line, max_length=300, truncate=True, padding=True, batched=False):
		""" This implements also BERT masks and type ids expected by BERT """
		if batched:
			return self.call_batched(line, max_length, truncate, padding)
		else:
			attention_mask = []
			type_ids = []
			out = {}
			out['input_ids'] = self.encode(line, max_length, truncate, padding)
			for token in out['input_ids']:
				if token != self.vocab['[PAD]']:
					attention_mask.append(1)
				else:
					attention_mask.append(0)
				type_ids.append(1) # default for non NSP
			out['attention_mask'] = attention_mask
			out['token_type_ids'] = type_ids
			return out


	def call_batched(self, batch_of_lines, max_length, truncate, padding):
		batch_out = []
		for line in batch_of_lines:
			batch_out.append(self.__call__(line, max_length, truncate, padding))
		return batch_out

