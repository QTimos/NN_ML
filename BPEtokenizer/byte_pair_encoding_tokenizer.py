#PAIRS_IDS = {}
#VOCAB_SIZE = 276
#def encode_byte_pair(pair, tokens):
#    global PAIRS_IDS
#    new_tokens = []
#    i = 0
#    try:
#        ids = max(PAIRS_IDS) + 1
#    except ValueError:
#        ids = 256
#    while i < len(tokens):
#        if i+1 < len(tokens) and (tokens[i], tokens[i+1]) == pair:
#            new_tokens.append(ids)
#            PAIRS_IDS[ids] = pair
#            i += 2
#        else:
#            new_tokens.append(tokens[i])
#            i += 1
#    return new_tokens
#
#def decode_byte_pair(Id, tokens):
#    global PAIRS_IDS
#    new_tokens = []
#    i = 0
#    while i < len(tokens):
#        if i+1 < len(tokens) and Id == tokens[i]:
#            new_tokens.append(PAIRS_IDS[Id][0])
#            new_tokens.append(PAIRS_IDS[Id][1])
#        else:
#            new_tokens.append(tokens[i])
#        i += 1
#    PAIRS_IDS = {k: v for k, v in PAIRS_IDS.items() if k != Id}
#    return new_tokens
#
#def get_byte_pairs_with_count(tokens):
#    pairs_with_count = {}
#    for p in zip(tokens, tokens[1:]):
#        pairs_with_count[p] = pairs_with_count.get(p, 0) + 1
#    return sorted([(v, k) for k, v in pairs_with_count.items()], reverse=True)
#
#
#def encode_text(txt: str):
#    tokens = list(map(int, txt.encode("utf-8")))
#    while True:
#        pairs_with_count = get_byte_pairs_with_count(tokens)
#        o, pair = pairs_with_count.pop(0)
#        if o <= 1:
#            break
#        tokens = encode_byte_pair(pair, tokens)
#    return tokens
#
#def decode_text(encoded_tokens: list[int]):
#    i = max(PAIRS_IDS)
#    tokens = encoded_tokens
#    while i > 255:
#        tokens = decode_byte_pair(i, tokens)
#        try:
#            i = max(PAIRS_IDS)
#        except ValueError:
#            break
#        txt = str(bytes(tokens).decode("utf-8", errors="replace"))
#        return txt
#
#def encode_to_max(txt: str):
#    tokens = list(map(int, txt.encode("utf-8")))
#    i = 0
#    max_comp = VOCAB_SIZE - 256
#    while i < max_comp:
#        pairs_with_count = get_byte_pairs_with_count(tokens)
#        o, pair = pairs_with_count.pop(0)
#        if o <= 1:
#            break
#        tokens = encode_byte_pair(pair, tokens)
#        i += 1
#    return tokens
#
#def encode(txt: str):
#    tokens = list(map(int, txt.encode("utf-8")))
#    while len(tokens) >= 2:
#        pairs_with_count = get_byte_pairs_with_count(tokens)
#        pair = min(pairs_with_count, key= lambda x: PAIRS_IDS.get(x, float("inf")))
#        if pair not in PAIRS_IDS:
#            break
#        tokens = encode_byte_pair(pair, tokens)
#    return tokens
#
#def main():
#    txt = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."""
#    encoded_str_tokens = encode_text(txt)
#
#    print(decode_text(encode("hello world")))
#
#
#if __name__ == "__main__":
#    main()

import json
import regex as re
from typing import Optional, List

class BPETokenizer:
    def __init__(self, vocab_size=276, vocab_path: Optional[str]=None):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {i: bytes([i]) for i in range(256)}
        self.pattern = "|".join([
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ])
        self.compiled_pattern = re.compile(self.pattern)
        if not vocab_path:
            self.vocab_path = "./vocab.json"
        else:
            self.vocab_path = vocab_path
            self._load_vocab()

    def _get_stats(self, chunks_tokens):
        counts = {}
        for tokens in chunks_tokens:
            for pair in zip(tokens, tokens[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text: str):
        text_chunks = self.compiled_pattern.findall(text)
        chunks_tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = self._get_stats(chunks_tokens)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] <= 1:
                break
            new_id = 256 + i
            self.encoder[best_pair] = new_id
            self.decoder[new_id] = self.decoder[best_pair[0]] + self.decoder[best_pair[1]]
            chunks_tokens = [self._merge_tokens(tokens, best_pair, new_id) for tokens in chunks_tokens]

    def _merge_tokens(self, tokens, pair, new_id):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode(self, text: str) -> List[int]:
        text_chunks = self.compiled_pattern.findall(text)
        final_tokens = []
        for chunk in text_chunks:
            chunk_tokens = list(chunk.encode("utf-8"))
            while len(chunk_tokens) >= 2:
                pairs = list(zip(chunk_tokens, chunk_tokens[1:]))
                valid_pairs = {pair: self.encoder[pair] for pair in pairs if pair in self.encoder}
                if not valid_pairs:
                    break
                best_pair = min(valid_pairs, key=valid_pairs.get)
                chunk_tokens = self._merge_tokens(chunk_tokens, best_pair, valid_pairs[best_pair])
            final_tokens.extend(chunk_tokens)
        return final_tokens

    def decode(self, tokens: List[int]) -> str:
        raw_bytes = b"".join(self.decoder[t] for t in tokens)
        return raw_bytes.decode("utf-8", errors="replace")

    def _load_vocab(self):
        with open(self.vocab_path, "r") as f:
            vocab = json.load(f)
        self.vocab_size = vocab["vocab_size"]
        encoder = vocab["encoder"]
        self.encoder = {tuple(map(int, k.split(","))): v for k, v in encoder.items()}
        for p, ids in sorted(self.encoder.items(), key=lambda x: x[1]):
            self.decoder[ids] = self.decoder[p[0]] + self.decoder[p[1]]

    def store_vocab(self):
        serializable_encoder = {f"{k[0]},{k[1]}": v for k, v in self.encoder.items()}
        vocab = {
            "vocab_size": self.vocab_size,
            "encoder": serializable_encoder
        }
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=4)


def test(tokenizer, test_str: str):
    encoded_tokens = tokenizer.encode(test_str)
    decoded_str = tokenizer.decode(encoded_tokens)
    print(f"Encoded tokens: {encoded_tokens}")
    print(f"Decoded string: {decoded_str}")

def main():
#    tokenizer = BPETokenizer(276, "./vocab.json")
    tokenizer = BPETokenizer(50000)
    string = """
The Project Gutenberg eBook of War and Peace, by Leo Tolstoy
This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online at
www.gutenberg.org. If you are not located in the United States, you
will have to check the laws of the country where you are located before
using this eBook.
“Well, Prince, so Genoa and Lucca are now just family estates of the
Buonapartes. But I warn you, if you don’t tell me that this means war,
if you still try to defend the infamies and horrors perpetrated by that
Antichrist—I really believe he is Antichrist—I will have nothing
more to do with you and you are no longer my friend, no longer my
‘faithful slave,’ as you call yourself! But how do you do? I see I
have frightened you—sit down and tell me all the news.”
It was in July, 1805, and the speaker was the well-known Anna Pávlovna
Schérer, maid of honor and favorite of the Empress Márya Fëdorovna.
With these words she greeted Prince Vasíli Kurágin, a man of high
rank and importance, who was the first to arrive at her reception. Anna
Pávlovna had had a cough for some days. She was, as she said, suffering
from la grippe; grippe being then a new word in St. Petersburg, used
only by the elite.
All her invitations without exception, written in French, and delivered
by a scarlet-liveried footman that morning, ran as follows:
“If you have nothing better to do, Count (or Prince), and if the
prospect of spending an evening with a poor invalid is not too terrible,
I shall be very charmed to see you tonight between 7 and 10—Annette
Schérer.”
“Heavens! what a virulent attack!” replied the prince, not in the
least disconcerted by this reception. He had just entered, wearing an
embroidered court uniform, knee breeches, and shoes, and had stars on
his breast and a serene expression on his flat face. He spoke in that
refined French in which our grandfathers not only spoke but thought, and
with the gentle, patronizing intonation natural to a man of importance
who had grown old in society and at court. He went up to Anna Pávlovna,
kissed her hand, presenting to her his bald, scented, and shining head,
and complacently seated himself on the sofa.
“First of all, dear friend, tell me how you are. Set your friend’s
mind at rest,” said he without altering his tone, beneath the
politeness and affected sympathy of which indifference and even irony
could be discerned.
“Can one be well while suffering morally? Can one be calm in times
like these if one has any feeling?” said Anna Pávlovna. “You are
staying the whole evening, I hope?”
“And the fete at the English ambassador’s? Today is Wednesday. I
must put in an appearance there,” said the prince. “My daughter is
coming for me to take me there.”
“I thought today’s fete had been canceled. I confess all these
festivities and fireworks are becoming wearisome.”
“If they had known that you wished it, the entertainment would have
been put off,” said the prince, who, like a wound-up clock, by force
of habit said things he did not even wish to be believed.
“Don’t tease! Well, and what has been decided about Novosíltsev’s
dispatch? You know everything.”
“What can one say about it?” replied the prince in a cold, listless
tone. “What has been decided? They have decided that Buonaparte has
burnt his boats, and I believe that we are ready to burn ours.”
Prince Vasíli always spoke languidly, like an actor repeating a stale
part. Anna Pávlovna Schérer on the contrary, despite her forty years,
overflowed with animation and impulsiveness. To be an enthusiast had
become her social vocation and, sometimes even when she did not
feel like it, she became enthusiastic in order not to disappoint the
expectations of those who knew her. The subdued smile which, though it
did not suit her faded features, always played round her lips expressed,
as in a spoiled child, a continual consciousness of her charming defect,
which she neither wished, nor could, nor considered it necessary, to
correct.
In the midst of a conversation on political matters Anna Pávlovna burst
out:
“Oh, don’t speak to me of Austria. Perhaps I don’t understand
things, but Austria never has wished, and does not wish, for war. She
is betraying us! Russia alone must save Europe. Our gracious sovereign
recognizes his high vocation and will be true to it. That is the one
thing I have faith in! Our good and wonderful sovereign has to perform
the noblest role on earth, and he is so virtuous and noble that God will
not forsake him. He will fulfill his vocation and crush the hydra of
revolution, which has become more terrible than ever in the person of
this murderer and villain! We alone must avenge the blood of the just
one.... Whom, I ask you, can we rely on?... England with her commercial
spirit will not and cannot understand the Emperor Alexander’s
loftiness of soul. She has refused to evacuate Malta. She wanted to
find, and still seeks, some secret motive in our actions. What answer
did Novosíltsev get? None. The English have not understood and cannot
understand the self-abnegation of our Emperor who wants nothing for
himself, but only desires the good of mankind. And what have they
promised? Nothing! And what little they have promised they will not
perform! Prussia has always declared that Buonaparte is invincible, and
that all Europe is powerless before him.... And I don’t believe a
word that Hardenburg says, or Haugwitz either. This famous Prussian
neutrality is just a trap. I have faith only in God and the lofty
destiny of our adored monarch. He will save Europe!”
She suddenly paused, smiling at her own impetuosity.
“I think,” said the prince with a smile, “that if you had been
sent instead of our dear Wintzingerode you would have captured the King
of Prussia’s consent by assault. You are so eloquent. Will you give me
a cup of tea?”
"""
    tokenizer.train(string)
    tokenizer.store_vocab()

#    print(re.findall(tokenizer.gpt2pattern, "Hello world14 HOW'RE you"))


if __name__ == "__main__":
    main()
