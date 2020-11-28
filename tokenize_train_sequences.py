##############################
# Tokenize sequences for BERT
##############################
import asmtokenizer
from asmtokenizer import ASMTokenizer
import json
import tqdm


def main():
	path_to_vocab = '/path/to/vocab/'
	path_to_sequences = '/path/to/train_sequences.txt'
	out = '/path/out.json'
	max_length = 200
	asmtokenizer = ASMTokenizer()
	asmtokenizer.load_pretrained(path_to_vocab)
	print('Processing...\n')
	with open(path_to_sequences, 'r') as to_tokenize:
		with open(out, 'w') as tokenized:
			for line in tqdm.tqdm(to_tokenize):
				if line == '\n':
					pass
				else:
					line_tokenized = json.dumps(asmtokenizer(line.strip(), max_length=max_length, truncate=True, padding=True))
					tokenized.write(line_tokenized)
					tokenized.write('\n')
	print('\nDone.\n')




if __name__ == '__main__':
	main()