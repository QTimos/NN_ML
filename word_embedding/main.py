import sys
import json
import numpy as np
from pathlib import Path
from typing import Generator, List, Union, Optional, Tuple
CONTEXT_WINDOW_SIZE = 2
VOCAB_SIZE = 5000
VOCAB_COUNTER = 0
EMBEDDING_DIM = 5
LEARNING_RATE = 0.75
VOCAB = {}
INPUT_GRIDM = (np.random.rand(1, EMBEDDING_DIM) - 0.5 ) * 0.1
OUTPUT_GRIDM = (np.random.rand(1, EMBEDDING_DIM) - 0.5 ) * 0.1
VOCAB_PATH = "./vocab.json"
EMBEDDINGS_PATH = "./embeddings.npz"

def usage() -> None:
    print("""
Usage:
    python3 main.py [option=path]
options:
    --dir_path:
        A directory with text files in it.
        exp:
            --dir_path=/home/user/textfiles
    --file_path:
        A valid file path.
        exp:
            --file_path=/home/user/textfile
            """)

def flatten(nested: List[Union[List[str], str]]) -> Generator:
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def has_punctuations(token: str, p: Optional[str]=None) -> bool:
    punctuations = ",.:;?!-_(){}[]\"'+=*/@#$%"
    if p is not None:
        for c in token:
            if c == p:
                return True
        return False
    for c in token:
        if c in punctuations:
            return True
    return False 

def split_on_puctuation(token: str) -> List[str]:
    punctuations = ",.:;?!-_(){}[]\"'+=*/@#$%"
    tokens = []
    i = 0
    while i < len(token):
        if has_punctuations(token, "@") or token.find("https://") != -1 or token.find("http://") != -1:
            tokens.append(token.strip("."))
            break
        if i < len(token) and token[i] in punctuations:
            if i-1 >= 0 and i+1 < len(token) and token[i] == "'":
                if not token[i-1].isspace() and not token[i+1].isspace():
                    ...
            elif token[i] == "#":
                if not token[i-1].isspace() and not token[i+1].isspace():
                   ... 
                if token[i-1].isspace() and not token[i+1].isspace():
                    idx = punctuations.index(token[i])
                    _, sep, word = token.strip().partition(punctuations[idx])
                    if len(f"{sep}{word}") > 0:
                        tokens.append(f"{sep}{word}")
                    token = token[i+len(word)+1::]
                    i = -1
            elif token[i] == ".":
                dig1, sep, dig2 = token.partition(".")
                if dig1.isdigit() and dig2.isdigit():
                    if len(f"{dig1}{sep}{dig2}") > 0:
                        tokens.append(f"{dig1}{sep}{dig2}")
                    token = token[i+len(dig2)+1::]
                    i = -1
                else:
                    idx = punctuations.index(token[i])
                    word1, sep, word2 = token.partition(punctuations[idx])
                    if len(word1) > 0:
                        tokens.append(word1)
                    if len(sep) > 0:
                        tokens.append(sep)
                    token = token[i+1::]
                    i = -1
            else:
                idx = punctuations.index(token[i])
                word1, sep, word2 = token.partition(punctuations[idx])
                if len(word1) > 0:
                    tokens.append(word1)
                if len(sep) > 0:
                    tokens.append(sep)
                token = token[i+1::]
                i = -1
        if (not has_punctuations(token) and len(token) > 0) or (has_punctuations(token) and len(token) == 1):
            tokens.append(token)
            break
        i += 1
    return tokens

def list_to_lower(l: List[str]) -> List[str]:
    for i in range(len(l)):
        l[i] = l[i].lower()
    return l

def tokenize(text: str) -> List[str]:
    tokens = text.strip().split()
    for i in range(len(tokens)):
        split_tokens = split_on_puctuation(tokens[i])
        if len(split_tokens) > 0:
            tokens[i] = split_tokens
    return list_to_lower(list(flatten(tokens)))

def get_pairs(tokens: List[str]) -> List[Tuple[str]]:
    pairs = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        j = CONTEXT_WINDOW_SIZE
        while j > 0:
            context = [word]
            if 0 <= i-j < len(tokens):
                context.append(tokens[i-j])
                if len(context) > 0 and (word, tokens[i-j]) not in pairs:
                    pairs.append(tuple(context))
            j -= 1
        j = 1
        while j <= CONTEXT_WINDOW_SIZE:
            context = [word]
            if 0 <= i+j < len(tokens):
                context.append(tokens[i+j])
                if len(context) > 0 and (word, tokens[i+j]) not in pairs:
                    pairs.append(tuple(context))
            j += 1
        i += 1
    return pairs

def word_to_id(word: str) -> int:
    return VOCAB[word]

def embed_word(pairs: List[Tuple[str]]) -> List[Tuple[int]]:
    id_pairs = []
    for p in pairs:
        id_pairs.append((word_to_id(p[0]), word_to_id(p[1])))
    return id_pairs

def add_to_vocab(tokens: List[str]) -> None:
    global VOCAB
    global VOCAB_COUNTER
    for t in tokens:
        if t not in VOCAB and VOCAB_COUNTER <= VOCAB_SIZE:
            VOCAB[t] = VOCAB_COUNTER
            VOCAB_COUNTER += 1
        elif VOCAB_COUNTER >= VOCAB_SIZE:
            raise ValueError("Vocab size exceeded!!")

def getting_the_probabilitiesV(idx: int) -> np.ndarray:
    centerV = INPUT_GRIDM[idx]
    dot_product = centerV @ OUTPUT_GRIDM.T
    dot_product -= np.max(dot_product)
    ind_exp = np.e**dot_product
    total_exp = np.sum(np.e**dot_product)
    probsV = ind_exp / total_exp
    return probsV

def getting_the_errorV(probsV: np.ndarray, pair: Tuple[int]) -> np.ndarray:
    errorsV = np.copy(probsV)
    errorsV[pair[1]] -= 1.0
    return errorsV

def learn_from_id_pairs(id_pairs: List[Tuple[int]]) -> None:
    global INPUT_GRIDM
    global OUTPUT_GRIDM
    for p in id_pairs:
        idx = p[0]
        probsV = getting_the_probabilitiesV(idx)
        errorsV = getting_the_errorV(probsV, p)
        output_weighted_gradientV = np.outer(errorsV, INPUT_GRIDM[idx])
        OUTPUT_GRIDM -= LEARNING_RATE * output_weighted_gradientV
        input_weighted_gradientV = errorsV @ OUTPUT_GRIDM
        INPUT_GRIDM[idx] -= LEARNING_RATE * input_weighted_gradientV

def store_vocab_and_weights() -> None:
    with open(VOCAB_PATH, "w") as f:
        json.dump(VOCAB, f, indent=4)
    np.savez(EMBEDDINGS_PATH, input_weights=INPUT_GRIDM, output_weights=OUTPUT_GRIDM)

def restore_vocab_and_weights(vocab_size: int) -> None:
    global VOCAB
    global VOCAB_COUNTER
    if Path(VOCAB_PATH).exists():
        with open(VOCAB_PATH, "r") as f:
            VOCAB = json.load(f)
            VOCAB_COUNTER = len(VOCAB.keys())
    global INPUT_GRIDM
    global OUTPUT_GRIDM
    if Path(EMBEDDINGS_PATH).exists():
        retrieved = np.load(EMBEDDINGS_PATH)
        if retrieved is None:
            raise FileNotFoundError("Embeddings file is not found or corrupted!!")
        INPUT_GRIDM = retrieved["input_weights"]
        OUTPUT_GRIDM = retrieved["output_weights"]
    else:
        global VOCAB_SIZE
        VOCAB_SIZE = vocab_size
        INPUT_GRIDM = (np.random.rand(VOCAB_SIZE, EMBEDDING_DIM) - 0.5 ) * 0.1
        OUTPUT_GRIDM = (np.random.rand(VOCAB_SIZE, EMBEDDING_DIM) - 0.5 ) * 0.1

def run_training(file_path) -> None:
    with open(file_path, "r") as f:
        data = f.read()
    tokens = tokenize(data)
    restore_vocab_and_weights(len(set(tokens)))
    str_pairs = get_pairs(tokens)
    add_to_vocab(tokens)
    id_pairs = embed_word(str_pairs)
    learn_from_id_pairs(id_pairs)
    store_vocab_and_weights()

def main() -> None:
    if len(sys.argv) != 2:
        print("Wrong amount of arguments.")
        usage()
        sys.exit(1)
    if "=" not in sys.argv[1]:
        print("Wrong input format.")
        usage()
        sys.exit(1)
    option, _, path = sys.argv[1].strip().partition("=")
    if len(option) == len(sys.argv[1]) and len(path) == 0:
        print("Wrong input format.")
        usage()
        sys.exit(1)
    if not Path(path).exists():
        print(f"Path '{path}' in input not found")
        usage()
        sys.exit(1)
    all_files = []
    if option == "--dir_path":
        for fp in Path(path).glob('**/*.txt'):
            all_files.append(fp)
    elif option == "--file_path":
        all_files.append(Path(path))
    else:
        print(f"Unknown option '{option}'")
        usage()
        sys.exit(1)
    if len(all_files) == 0:
        print("No text files found to process.")
        sys.exit(0)
    print("Phase 1: Analyzing files and building global vocabulary...")
    for file_path in all_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
            tokens = tokenize(data)
            add_to_vocab(tokens)
        except Exception as e:
            print(f"Skipping file {file_path} due to read error: {e}")
    print(f"Phase 2: Setting up matrices for vocabulary size: {len(VOCAB)}")
    restore_vocab_and_weights(len(VOCAB))
    print("Phase 3: Starting training passes...")
    for file_path in all_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read()
            tokens = tokenize(data)
            str_pairs = get_pairs(tokens)
            id_pairs = embed_word(str_pairs)
            learn_from_id_pairs(id_pairs)
            print(f" Trained successfully on: {file_path.name}")
        except Exception as e:
            print(f"Skipping training pass for {file_path.name}: {e}")
    print("Phase 4: Archiving models and vocabulary maps to storage...")
    store_vocab_and_weights()
    print("Training job complete!")



if __name__ == "__main__":
    main()
