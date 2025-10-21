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

print(N[0])