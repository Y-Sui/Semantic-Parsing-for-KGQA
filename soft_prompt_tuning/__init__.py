import torch
import torch.nn as nn


class SoftEmbedding(nn.Module):
    def __init__(self, wte=nn.Embedding, n_tokens=10, random_range=0.5, initialize_from_vocab=True):
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initialize from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.random_range = random_range
        self.initialize_from_vocab = initialize_from_vocab
        parameters = self.initialize_embedding()
        self.learned_embedding = nn.parameter.Parameter(parameters)

    def initialize_embedding(self):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if self.initialize_from_vocab:
            return self.wte.weight[:self.n_tokens].clone().detach()
        return torch.tensor(self.n_tokens, self.wte.weight.size(1)).to(torch.float).uniform_(-self.random_range, self.random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specific embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)