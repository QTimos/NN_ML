import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import Generator, List, Union, Optional, Tuple
CONTEXT_WINDOW_SIZE = 2
VOCAB_SIZE = 50000
MAX_PAIRS = 500000
NEG_SAMPLES = 5
VOCAB_COUNTER = 0
EMBEDDING_DIM = 5
LEARNING_RATE = 0.01
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
    seen = set()
    for i, word in enumerate(tokens):
        for j in range(1, CONTEXT_WINDOW_SIZE + 1):
            for neighbor_i in (i - j, i + j):
                if 0 <= neighbor_i < len(tokens):
                    pair = (word, tokens[neighbor_i])
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)
    return pairs

def word_to_id(word: str) -> int:
    return VOCAB[word]

def embed_word(pairs: List[Tuple[str]]) -> List[Tuple[int]]:
    id_pairs = []
    for p in pairs:
        if p[0] in VOCAB and p[1] in VOCAB:
            id_pairs.append((word_to_id(p[0]), word_to_id(p[1])))
    return id_pairs

def add_to_vocab(tokens: List[str]) -> None:
    global VOCAB, VOCAB_COUNTER
    for t in tokens:
        if t not in VOCAB and VOCAB_COUNTER < VOCAB_SIZE:
            VOCAB[t] = VOCAB_COUNTER
            VOCAB_COUNTER += 1

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
    global INPUT_GRIDM, OUTPUT_GRIDM
    vocab_size = INPUT_GRIDM.shape[0]
    pairs = np.array(id_pairs)
    center_ids = pairs[:, 0]
    ctx_ids = pairs[:, 1]
    center_vecs = INPUT_GRIDM[center_ids]
    ctx_vecs = OUTPUT_GRIDM[ctx_ids]
    pos_scores = np.sum(center_vecs * ctx_vecs, axis=1)
    pos_sig = 1.0 / (1.0 + np.exp(-np.clip(pos_scores, -10, 10)))
    pos_err = pos_sig - 1.0
    neg_ids = np.random.randint(0, vocab_size, size=(len(id_pairs), NEG_SAMPLES))
    neg_vecs = OUTPUT_GRIDM[neg_ids]
    neg_scores = np.einsum('nd,nkd->nk', center_vecs, neg_vecs)
    neg_sig = 1.0 / (1.0 + np.exp(-np.clip(neg_scores, -10, 10)))
    pos_grad_in = pos_err[:, None] * ctx_vecs
    neg_grad_in = np.einsum('nk,nkd->nd', neg_sig, neg_vecs)
    input_grad = pos_grad_in + neg_grad_in
    pos_grad_out = pos_err[:, None] * center_vecs
    neg_grad_out = neg_sig[:, :, None] * center_vecs[:, None, :]
    np.add.at(INPUT_GRIDM, center_ids, -LEARNING_RATE * input_grad)
    np.add.at(OUTPUT_GRIDM, ctx_ids, -LEARNING_RATE * pos_grad_out)
    np.add.at(OUTPUT_GRIDM, neg_ids.ravel(), -LEARNING_RATE * neg_grad_out.reshape(-1, OUTPUT_GRIDM.shape[1]))

def test(string: str) -> List[str]:
    tokens = tokenize(string)
    results = []
    for word in tokens:
        if word not in VOCAB:
            results.append(f"'{word}' not in vocabulary")
            continue
        idx = VOCAB[word]
        word_vec = INPUT_GRIDM[idx]
        norms = np.linalg.norm(INPUT_GRIDM, axis=1)
        word_norm = np.linalg.norm(word_vec)
        sims = (INPUT_GRIDM @ word_vec) / (norms * word_norm + 1e-8)
        sims[idx] = -1
        top_ids = np.argsort(sims)[::-1][:10]
        reverse_vocab = {v: k for k, v in VOCAB.items()}
        neighbors = [f"{reverse_vocab[i]} ({sims[i]:.3f})" for i in top_ids]
        results.append(f"'{word}': {', '.join(neighbors)}")
    return results

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
            if len(id_pairs) > MAX_PAIRS:
                random.shuffle(id_pairs)
                id_pairs = id_pairs[:MAX_PAIRS]
            learn_from_id_pairs(id_pairs)
            print(f" Trained successfully on: {file_path.name}")
        except Exception as e:
            print(f"Skipping training pass for {file_path.name}: {e}")
    print("Phase 4: Archiving models and vocabulary maps to storage...")
    store_vocab_and_weights()
    print("Training job complete!")
    print("Testing!")
    string = str(input("Enter your prompt: "))
    print(test(string))



if __name__ == "__main__":
    main()
