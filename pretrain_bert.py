################################
# Pretrain a new BERT Model
################################
import transformers
import datasets
from datasets import load_dataset
import asmtokenizer
from asmtokenizer import ASMTokenizer
import personaldatacollator
from personaldatacollator import PersonalDataCollator
from transformers import Trainer, TrainingArguments, BertConfig, BertForMaskedLM
import numpy as np
import logging
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def get_tokenizer(path_to_vocab):
	asmtokenizer = ASMTokenizer()
	asmtokenizer.load_pretrained(path_to_vocab)
	return asmtokenizer


def get_datasets(train_tokenized, eval_tokenized):
	train_dataset = load_dataset('json', data_files=train_tokenized, split='train')
	eval_dataset = load_dataset('json', data_files=eval_tokenized, split='train')
	train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])
	eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])
	return train_dataset, eval_dataset


def get_data_collator(asmtokenizer):
	data_collator = PersonalDataCollator(tokenizer=asmtokenizer, mlm=True, mlm_probability=0.15)
	return data_collator


def flat_accuracy(preds, labels):
	pred_flat = preds.flatten()
	labels_flat = labels.flatten()
	true_labels = []
	true_preds = []
	for ix in range(labels_flat.shape[0]):
		if labels_flat[ix] != -100:
			true_labels.append(labels_flat[ix])
			true_preds.append(pred_flat[ix])
	true_labels = np.asarray(true_labels)
	true_preds = np.asarray(true_preds)
	num = np.sum(true_preds == true_labels)
	den = len(true_labels)
	acc = num / den
	return acc


def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)
	score = flat_accuracy(preds, labels)
	return {'accuracy': score}


def setup_bert(size):
	config = BertConfig(
	vocab_size=size,
	hidden_size=768,
	output_hidden_states=False,
	max_position_embeddings=250,
	intermediate_size=3072,
	num_attention_heads=8,
	num_hidden_layers=8,
	type_vocab_size=1,
	)
	model = BertForMaskedLM(config)
	return model


def setup_trainer(out_dir, bert_model, data_collator, train_dataset, eval_dataset, compute_metrics):
	training_args = TrainingArguments(output_dir=out_dir, overwrite_output_dir=True,
		do_train=True,
		do_eval=True,
		evaluation_strategy='epoch',
		num_train_epochs=6,
		per_device_train_batch_size=32,
		save_steps=10_000,
		save_total_limit=20
	)

	trainer = Trainer(
		model=bert_model,
		args=training_args,
		data_collator=data_collator,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
	)
	return trainer


def _mp_fn(index):
	train()


def train():
	out_dir = '/path/to/results/out_dir/'
	path_to_vocab = '/path_to_vocab/'
	train_tokenized = '/path/to/train/tokenized/'
	eval_tokenized = '/path/to/eval/tokenized/'
	asmtokenizer = get_tokenizer(path_to_vocab)
	train_dataset, eval_dataset = get_datasets(train_tokenized, eval_tokenized)
	data_collator = get_data_collator(asmtokenizer)
	bert_model = setup_bert(asmtokenizer.get_vocab_length())
	trainer = setup_trainer(bert_model, data_collator, train_dataset, eval_dataset, compute_metrics)
	logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if xm.get_ordinal() == 0 else logging.WARNING
	)
	transformers.utils.logging.set_verbosity_info()
	trainer.train()
	trainer.save_model(out_dir+'model/')
	trainer.evaluate()
	return


def main():
	xmp.spawn(_mp_fn, nprocs=1, start_method='fork')
	print('\nDone.\n')
	return


if __name__ == '__main__':
	main()