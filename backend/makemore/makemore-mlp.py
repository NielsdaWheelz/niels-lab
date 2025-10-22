from rich import print
import torch
import torch.nn.functional as F
from pathlib import Path
names_path = Path(__file__).with_name("names.txt")
names = names_path.read_text().splitlines()

characters = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(characters)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

generator = torch.Generator().manual_seed(2147483647)

block_size = 3 # context length, number of characters to look at to predict the next character
X, Y = [], [] # inputs and labels

for name in names:
  context = [0] * block_size # start with a context of all 0s
  for char in name + '.': # iterate over the characters in the name. padding with '.' to the end
    index = stoi[char] # get the index of the character
    X.append(context) # add the context to the inputs
    Y.append(index) # add the index to the labels
    context = context[1:] + [index] # shift the context to the right
X = torch.tensor(X)
Y = torch.tensor(Y)

# each character will have a 2-dimensional embedding
C = torch.randn((27, 2), generator = generator)
# embedding_for_e = C[5] # embedding for the character 'e'
# one_hot_for_e = F.one_hot(torch.tensor([5]), num_classes = 27).float() # one-hot encoding for the character 'e'
# embedding_for_e == one_hot_for_e @ C # embedding for the character 'e'
W1 = torch.randn((6, 100), generator = generator) # 3x2=6 is the number of neurons in the first layer, 100 is the number of neurons in the second layer
b1 = torch.randn((100), generator = generator) # bias for the first layer
W2 = torch.randn((100, 27), generator = generator) # 100 is the number of neurons in the second layer, 27 is the number of characters
b2 = torch.randn(27, generator = generator) # bias for the second layer
parameters = [C, W1, b1, W2, b2]

for p in parameters:
  p.requires_grad = True

for k in range(10):
  # create mini-batches
  index = torch.randint(0, X.shape[0], (32,)) # index is a tensor of shape (32,), where the first dimension is the number of examples, the second dimension is the index of the example
  Xb = X[index] # Xb is a tensor of shape (32, 3), where the first dimension is the number of examples, the second dimension is the index of the example
  Yb = Y[index] # Yb is a tensor of shape (32,), where the first dimension is the number of examples, the second dimension is the index of the example
  # forward pass
  embeddings = C[Xb] # embeddings for the input context. Xb(32, 3)x2(2 embedding dimensions)
  # .view(-1, 6) is to reshape the embeddings tensor to a 2D tensor, 
  # where the first dimension is inferred from the shape of the other dimensions (-1), 
  # 6 is the number of neurons in the first layer
  hidden_layer_activations = torch.tanh(embeddings.view(-1, 6) @ W1 + b1)
  logits = hidden_layer_activations @ W2 + b2
  # counts = logits.exp()
  # probabilities = counts / counts.sum(1, keepdim = True)
  # probabilities = F.softmax(logits, dim=1)
  # probabilities is a tensor of shape (32, 27), where the first dimension is the number of examples, the second dimension is the number of characters
  # loss = -probabilities[torch.arange(32), Y].log().mean()
  loss = F.cross_entropy(logits, Yb) # another way to calculate the loss

  # reset the gradients to 0
  for p in parameters:
    p.grad = None

  # backward pass
  loss.backward()

  # update the weights
  for p in parameters:
    p.data += -0.1 * p.grad
  print(f"step {k}: loss {loss.item()}")