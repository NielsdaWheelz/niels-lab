from rich import print
import torch
import matplotlib.pyplot as plt
from pathlib import Path
# open('names.txt') in backend/makemore/makemore.py:1 looks in whatever directory you launch the script from. 
# When you run it from the repo root, there is no names.txt, so Python raises a FileNotFoundError.
names_path = Path(__file__).with_name("names.txt")
names = names_path.read_text().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

characters = sorted(list(set(''.join(names))))
string_to_int = {string:integer+1 for integer, string in enumerate(characters)}
string_to_int['.'] = 0
int_to_string = {integer:string for string, integer in string_to_int.items()}

# char_counts = sorted(b.items(), key = lambda key_value: key_value[1], reverse=True)

# count the bigrams in the names
for n in names:
  chars = ['.'] + list(n) + ['.']
  # zip returns tuples
  # but if one array is shorter, zip will stop at the end of the shorter array
  for char1, char2 in zip(chars, chars[1:]):
    integer_1 = string_to_int[char1]
    integer_2 = string_to_int[char2]
    bigram = (integer_1, integer_2)
    N[integer_1, integer_2] += 1
    # print(char1, char2)

# print(int_to_string)

plt.figure(figsize=(16,16))
plt.imshow(N, cmap="Blues")
for i in range(27):
  for j in range(27):
    char_string = int_to_string[i] + int_to_string[j]
    plt.text(j, i, char_string, ha="center", va="bottom", color="gray")
    plt.text(j, i, N[i,j].item(), ha="center", va="top", color="gray")
# plt.axis("off")
# plt.show()

# print(N[0])

# probability of a given first character
# turn the row into a probability distribution by dividing by the sum of the row
p = N[0].float() # turn the row into a float tensor
p = p / p.sum()


g = torch.Generator().manual_seed(2147483647) # seed the generator so i match karpathy
# p = torch.rand(3, generator=g) # generate 3 random numbers from the probability distribution
# p = p / p.sum()

# use the multinomial to draw samples from it
multinomial = torch.multinomial(p, num_samples=1, replacement=True, generator=g)
# letters = [int_to_string[i.item()] for i in multinomial]
# print(''.join(letters))
letter = int_to_string[multinomial.item()]
# print(letter)

# probabilities; basically our parameters
P = N.float()
P /= P.sum(1, keepdim=True) # divide each row by the sum of the row to get a probability distribution
print(P)

for i in range(10):
  out = []
  index = 0
  while True:
    p = P[index]
    # p = N[index].float() # turn the row into a float tensor
    # p = p / p.sum() # turn the row into a probability distribution
    index = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # draw a sample from the probability distribution
    out.append(int_to_string[index])
    if index == 0: # if the sample is 0, we have reached the end of the name
      break
  # print(''.join(out))

for n in names[:3]:
  chars = ['.'] + list(n) + ['.']
  # zip returns tuples
  # but if one array is shorter, zip will stop at the end of the shorter array
  log_likelihood = 0.0
  n = 0
  for char1, char2 in zip(chars, chars[1:]):
    integer_1 = string_to_int[char1]
    integer_2 = string_to_int[char2]
    p = P[integer_1, integer_2]
    log_p = torch.log(p)
    log_likelihood += log_p
    n += 1
    # print(char1, char2, p.item(), log_p.item())
print(f'{-log_likelihood/n=}')
# -log_likelihood/n is the average log probability of the characters in the name