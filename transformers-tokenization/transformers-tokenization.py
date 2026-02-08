import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<UNK>"] = 1
        self.word_to_id["<BOS>"] = 2
        self.word_to_id["<EOS>"] = 3

        curr_index = 4
        all_words = set()
        
        for sentence in texts:
            text_split = sentence.split()
            for word in text_split:
                if word not in all_words:
                    all_words.add(word)

        for word in all_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = curr_index
                curr_index+=1
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.id_to_word)

    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        token_ids =  [self.word_to_id.get(word, self.word_to_id["<UNK>"])  for word in text.split()]

        return token_ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return " ".join([self.id_to_word[index] for index in ids])
