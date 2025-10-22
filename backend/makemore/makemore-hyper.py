from rich import print
import torch
import torch.nn.functional as F
from pathlib import Path
import random

names_path = Path(__file__).with_name("names.txt")
names = names_path.read_text().splitlines()

characters = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(characters)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

generator = torch.Generator().manual_seed(2147483647)

def build_dataset(names):
  block_size = 3
  X, Y = [], []
  for name in names:
    context = [0] * block_size
    for char in name + '.':
      index = stoi[char]
      X.append(context)
      Y.append(index)
      context = context[1:] + [index]
  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

random.seed(42)
random.shuffle(names)
n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))
Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte, Yte = build_dataset(names[n2:])

C = torch.randn((27, 2), generator = generator)
W1 = torch.randn((6, 100), generator = generator)
b1 = torch.randn((100), generator = generator)
W2 = torch.randn((100, 27), generator = generator)
b2 = torch.randn(27, generator = generator)
parameters = [C, W1, b1, W2, b2]

n_params = sum(p.numel() for p in parameters)
print(f"Number of parameters: {n_params}")


for p in parameters:
  p.requires_grad = True

for k in range(10000):
  # construct a mini-batch
  index = torch.randint(0, Xtr.shape[0], (32,)) # (32,) a tensor of 32 random integers between 0 and the number of examples in the training set

  # forward pass
  Xb = Xtr[index] # (32, 3) selects a tensor of 32 examples, each with a context of 3 characters
  Yb = Ytr[index] # (32,) selects a tensor of 32 labels, each corresponding to the correct next character
  embeddings = C[Xb] # (32, 3, 2) embeds the context of 3 characters into a 2-dimensional vector
  hidden_layer_activations = torch.tanh(embeddings.view(-1, 6) @ W1 + b1) # (32, 100) applies the weights and biases to the embeddings to get the hidden layer activations
  logits = hidden_layer_activations @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Yb)

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update the weights
  learning_rate = 0.01
  for p in parameters:
    p.data += -learning_rate * p.grad
print("training loss", loss.item())

# dev set evaluation
embeddings = C[Xdev] # (32, 3, 2)
h = torch.tanh(embeddings.view(-1, 6) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev) # (32,)
print("dev set evaluation", loss.item())

# test set evaluation
embeddings = C[Xte] # (32, 3, 2)
h = torch.tanh(embeddings.view(-1, 6) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Yte) # (32,)
print("test set evaluation", loss.item())