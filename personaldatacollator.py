#################################
# PersonalDataCollator for MLM
#################################
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from dataclasses import dataclass


@dataclass
class PersonalDataCollator:
    """
    Data collator used for masked language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: ASMTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = self._tensorize_batch([e['input_ids'] for e in examples], self.tokenizer.vocab['[PAD]'])
        token_type_ids = self._tensorize_batch([e['token_type_ids'] for e in examples], 0)
        attention_mask = self._tensorize_batch([e['attention_mask'] for e in examples], 0)

        if self.mlm:
            input_ids, labels = self.mask_tokens(input_ids)
        else:
            labels = input_ids.clone().detach()
            labels[labels == self.vocab['[PAD]']] = -100

        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "labels": labels}

    def _tensorize_batch(self, batch: List[torch.Tensor], pad_token: int) -> torch.Tensor:
        length_of_first = batch[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in batch)
        if are_tensors_same_length:
            return torch.stack(batch, dim=0)
        else:
          raise ValueError("Sequences in batches not all of the same size.")
            

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        padding_mask = labels.eq(self.tokenizer.vocab['[PAD]'])
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.vocab['[MASK]']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels