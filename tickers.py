import os

file_path = os.path.join(os.path.dirname(__file__), 'tickers.txt')

with open(file_path, 'w') as f:
    for i in range(600000, 604000):
        f.write(f"{i}.SS\n")
    for i in range(688000, 688999):
        f.write(f"{i}.SS\n")
    for i in range(900000, 900999):
        f.write(f"{i}.SS\n")

print("done")
