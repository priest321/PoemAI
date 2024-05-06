import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import requests
import pandas as pd
import math
import json
import random
import matplotlib.pyplot as plt

FILE_PATH = "data/music_training_data.txt"
MODEL_PATH = "pre_trained_model/LLM_poem.pth"

# hyperparamemters
BATCH_SIZE = 4
CTX_LENGTH = 64
NUM_BLOCK = 8
D_MODEL = 64 # should be 512
NUM_HEAD = 4
HEAD_SIZE = int(D_MODEL/NUM_HEAD)
LR = 0.001
DROP_OUT = 0.2
EVAL_ITERS = 10 # Because the loss.backward this num should not be too big
VALID_EVAL_ITERS = 5
EVAL_MAX = 2000
PUNCTUATION = [",", ".", "!", ":", "!", "\n"]
TEXT = []
TEMPERATURE = 1.0
TORCH_SEED = 1337
VALID_INPUT = ["我", "伤", "责", "脆"]
SENTENCE_INPUT = "悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月悠悠岁月"
torch.manual_seed(TORCH_SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


with open(FILE_PATH, 'r', encoding="utf8") as f:
    text = f.read()
    print(len(text))


class FeedforwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(DROP_OUT)
        )
    def forward(self, x):
        return self.forward_model(x)
        

class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        #reference word from transformer
        self.Wq = nn.Linear(D_MODEL, HEAD_SIZE, bias=False)
        self.Wk = nn.Linear(D_MODEL, HEAD_SIZE, bias=False)
        self.Wv = nn.Linear(D_MODEL, HEAD_SIZE, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CTX_LENGTH, CTX_LENGTH)))
        self.dropout = nn.Dropout(DROP_OUT)
        
    def forward(self, x):
        batch_size, current_ctx, dimension = x.shape
        if current_ctx <= CTX_LENGTH and dimension == D_MODEL:
            
            Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
            weights = (Q @ K.transpose(-2, -1)) * (1.0/math.sqrt(K.size(-1)))
            weights = weights.masked_fill(self.tril[:current_ctx, :current_ctx] == 0, float('-inf')) # check sytex
            weights = F.softmax(weights, dim=-1)
            weights = self.dropout(weights)
            output = weights @ V
            return output
        else:
            raise ValueError(f"Invalid input shape: {x.shape} value: {x}")


class MultiHeadModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionModule() for head in range(NUM_HEAD)])
        self.Wo_layer = nn.Linear(D_MODEL, D_MODEL)
        self.dropout = nn.Dropout(DROP_OUT)
        
    def forward(self, x):
        # reshape multihead back to original shape
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.Wo_layer(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(D_MODEL)
        self.layer_norm2 = nn.LayerNorm(D_MODEL)
        self.multi_head_attention_layer = MultiHeadModule()
        self.feedforward_network = FeedforwardNetwork(D_MODEL)
        
    def forward(self, x):
        """ 
        1. residual connection 1: add x back after layer normalization -> multihead attentions-> dropout  
        2. residual connection 2: add x_weight back after layer normalization -> feed forward -> dropout
        """
        x_norm_1 = self.layer_norm1(x)
        x_weight = x + self.multi_head_attention_layer(x_norm_1)
        x_norm_2 = self.layer_norm2(x_weight)
        return x_weight + self.feedforward_network(x_norm_2)

    
class LLMModel(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.index_to_word = {}
        self.word_to_index = {PUNCTUATION[i]: i for i in range(len(PUNCTUATION))}
        self.data = []
        self.tokenized_text = []
        self.max_token_value = None
        self.token_embedding_table = None
        # unpack list of transformer blocks and layerNorm
        self.transformer_blocks = nn.Sequential(*(
                        [TransformerBlock() for block in range(NUM_BLOCK)]+[nn.LayerNorm(D_MODEL)]))
        
        self.embedding_text(text)
        self.get_token_embedding_table()
        
        self.dimontion_to_all_words_layer = nn.Linear(D_MODEL, self.max_token_value)
    
    def get_token_embedding_table(self):
        tokenized_text = torch.tensor(self.tokenized_text+1, dtype=torch.long, device=DEVICE)
        self.max_token_value = tokenized_text.max().item()
        self.token_embedding_table = nn.Embedding(self.max_token_value+1, D_MODEL)
        
    def embedding_text(self, text):
        count = len(PUNCTUATION)
        for c in text:
            if c not in self.word_to_index:
                count += 1
                self.word_to_index[c] = count
            # create the entire data with good saparation
            self.data.append(c)
                    
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.tokenized_text = torch.tensor([self.word_to_index[i] for i in self.data], dtype=torch.long, device=DEVICE)

    def word_position_embedding(self, idx):
        # idx id of input x value
        batch, current_cxt_len = idx.shape
        #print("debug:", int(current_cxt_len)) # 16
        position_encoding_matrix = torch.zeros(CTX_LENGTH, D_MODEL)
        position_tensor = torch.arange(0, CTX_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2).float()*(-math.log(10000.0) / D_MODEL))
        position_encoding_matrix[:, 0::2] = torch.sin(position_tensor*div_term)
        position_encoding_matrix[:, 1::2] = torch.cos(position_tensor*div_term)
        # print("position_matrix: ", position_encoding_matrix.shape)  torch.Size([16, 64]
        position_embedding = position_encoding_matrix[:current_cxt_len, :].to(DEVICE)
        # print("position_embedding: ", position_embedding.shape)  torch.Size([16, 64]
        return position_embedding
        
    def forward(self, idx, targets=None):
        loss = None
        position_embedding = self.word_position_embedding(idx)
        x = self.token_embedding_table(idx) + position_embedding
        x_output = self.transformer_blocks(x)
        
        logits = self.dimontion_to_all_words_layer(x_output)
        
        if targets != None:
            batch, ctx_len, max_token_len = logits.shape
            logits_reshaped = logits.reshape(batch*ctx_len, max_token_len)
            targets_reshaped = targets.reshape(batch*ctx_len)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)

        return logits, loss
    
    def get_next_word(self, idx):
        logits, loss = self.forward(idx)
        probs = F.softmax(input=logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1)
        # idx_next_sample = torch.multinomial(probs[0][0], 1).item()
        idx = torch.tensor([[idx_next[-1][-1].item()]], dtype=torch.long, device=DEVICE)
        return idx

    def process_word(self, idx, max_new_tokens):
        output = [idx.item()]
        for token in range(max_new_tokens):
            idx = self.get_next_word(idx)
            output.append(idx.item())
        
        return " ".join([self.index_to_word[index] for index in output])

    def process_sentence(self, idx, max_new_tokens, ctx_len):
        output = [c.item() for c in idx[0]]
        for token in range(max_new_tokens):
            idx = idx[:,-CTX_LENGTH:] if ctx_len > CTX_LENGTH else idx
            idx_next = self.get_next_word(idx)
            idx = torch.cat((idx, idx_next), dim=-1)
            output.append(idx_next.item())
        
        return " ".join([self.index_to_word[index] for index in output])

    def generate(self, idx, max_new_tokens=20):
        # check shape
        batch, text = idx.shape
        return self.process_word(idx, max_new_tokens) if text==1 else self.process_sentence(idx, max_new_tokens, text)


def get_batch(data_type: str, train_data, test_data):
    data = train_data if data_type == "train" else test_data
    idxs = torch.randint(low=0, high=len(data) - CTX_LENGTH, size=(BATCH_SIZE,))
    return torch.stack([data[idx:idx+CTX_LENGTH] for idx in idxs]).to(DEVICE), torch.stack([data[idx+1:idx+CTX_LENGTH+1] for idx in idxs]).to(DEVICE)


@torch.no_grad()
def estimate_loss(LLM_model, train_data, test_data):
    output = {"train": [], "valid": []}
    
    # Disable learning
    LLM_model.eval()
    for data_type in ["train", "valid"]:
        losses = torch.zeros(VALID_EVAL_ITERS)
        for k in range(VALID_EVAL_ITERS):
            x_batch, y_batch = get_batch(data_type, train_data, test_data)
            logits, loss = LLM_model(x_batch, y_batch)
            logits
            losses[k] = loss.item()
        output[data_type] = losses.mean()
    
    # Single world test
    if VALID_INPUT:
        for c in VALID_INPUT:
            test_input = torch.tensor([[LLM_model.word_to_index[c]]], dtype=torch.long, device=DEVICE)
            print(LLM_model.generate(test_input, 10))
    
    # Sentence test
    if SENTENCE_INPUT:
        test_sentence_input = torch.tensor([[LLM_model.word_to_index[c] for c in list(SENTENCE_INPUT)]], dtype=torch.long, device=DEVICE)
        print(LLM_model.generate(test_sentence_input, 10))
        
    # Active learning
    LLM_model.train()
    return output


def display_graph(loss_history):
    plt.figure()
    plt.subplot(1, 2, 1)
    for i in loss_history:
        plt.plot([round(i.get("train").item(), 3) for i in loss_history], label="Training Loss")
        plt.xlabel('Batch')
        plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    for i in loss_history:
        plt.plot([round(i.get("valid").item(), 3) for i in loss_history], label="Validion Loss")
        plt.xlabel('Batch')
        plt.ylabel('Loss')
    plt.title("Validation Loss")
    plt.show()


def get_LLM_model():
    model = LLMModel(text)
    print("max_token_value: ", model.max_token_value)
    separate_index = int(len(model.tokenized_text)*0.9)
    train_data = model.tokenized_text[:separate_index]
    test_data = model.tokenized_text[separate_index:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    track_losses = []

    for step in range(EVAL_MAX):
        if step % EVAL_ITERS == 0 or (step == EVAL_MAX -1):
            e_loss = estimate_loss(model, train_data, test_data)
            track_losses.append(e_loss)
            print("steps", step, "loss", round(e_loss["train"].item(), 3), "validation loss: ", round(e_loss['valid'].item(), 3))
        xb, yb = get_batch('train', train_data, test_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model, MODEL_PATH)
    
    #display_graph(track_losses)

if __name__ == "__main__":
    get_LLM_model()

