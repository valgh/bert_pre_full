######################
# Train ASM Tokenizer
######################
import os
import asmtokenizer
from asmtokenizer import ASMTokenizer


def main():
	path_to_train_file = '/path/to/train_sequences.txt'
	asmtokenizer = ASMTokenizer()
	# or multiprocessing=True, pools_default=10
	asmtokenizer.train(path_to_train_file, min_frequency=8, max_tokens=32000, multiprocessing=False)
	print('Done.')


if __name__ == '__main__':
	main()