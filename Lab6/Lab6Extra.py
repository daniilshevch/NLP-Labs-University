import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

MESSAGE = "за останні три роки я проїхав 180000 км на потягах укрзалізниці"

EMBEDDING_DIM = 48
HIDDEN_DIM = 160
START, END = "[", "]"

MAX_EPOCHS = 5000
LR = 3e-3
CLIP = 1.0
SEED = 42
DECIMALS = 4


class ElmanRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.W_h = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM, bias=True)    # weight: (160, 48)
        self.U_h = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)      # weight: (160, 160)
        self.W_y = nn.Linear(HIDDEN_DIM, vocab_size, bias=False)      # weight: (133, 160)

    def forward(self, indices):
        h = torch.zeros(HIDDEN_DIM)
        logits = []
        for x in self.embedding(indices):
            h = torch.relu(self.W_h(x) + self.U_h(h))
            logits.append(self.W_y(h))
        return torch.stack(logits)


def greedy_decode(emb, W_h, b, U_h, W_y, vocab, max_steps=300):
    char_to_index = {char: index for index, char in enumerate(vocab)}
    current_index = char_to_index[START]
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    out = []
    for _ in range(max_steps):
        h = np.maximum(0, W_h @ emb[current_index] + b + U_h @ h)
        y = W_y @ h
        current_index = int(np.argmax(y))
        if vocab[current_index] == END:
            return "".join(out), True
        out.append(vocab[current_index])
    return "".join(out), False


def weights_numpy(model, decimals=None):
    w = {
        "embedding.weight": model.embedding.weight,
        "W_h.weight": model.W_h.weight,
        "W_h.bias": model.W_h.bias,
        "U_h.weight": model.U_h.weight,
        "W_y.weight": model.W_y.weight,
    }
    w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in w.items()}
    if decimals is not None:
        w = {k: np.round(v, decimals) for k, v in w.items()}
    return w


def decode_with(w, vocab):
    return greedy_decode(w["embedding.weight"], w["W_h.weight"], w["W_h.bias"],
                         w["U_h.weight"], w["W_y.weight"], vocab)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", default="vocab.json")
    p.add_argument("--out", default="weights")
    p.add_argument("--message", default=MESSAGE)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)
    char_to_index = {c: i for i, c in enumerate(vocab)}

    missing = set(args.message) - set(vocab)
    if missing:
        raise SystemExit(f"Цих символів немає у словнику: {sorted(missing)}")
    if START in args.message or END in args.message:
        raise SystemExit("Повідомлення не може містити '[' або ']'")

    seq = [char_to_index[START]] + [char_to_index[c] for c in args.message] + [char_to_index[END]]
    inputs = torch.tensor(seq[:-1])
    targets = torch.tensor(seq[1:])

    model = ElmanRNN(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, MAX_EPOCHS + 1):
        opt.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()

        if epoch % 50 == 0:
            text, ended = decode_with(weights_numpy(model, DECIMALS), vocab)
            print(f"epoch {epoch:5d}  loss {loss.item():.5f}  -> {text[:70]!r}")

            if ended and text == args.message and loss.item() < 0.01:
                break
    else:
        print("Увага: не вдалося досягти точного відтворення. Збільште MAX_EPOCHS або змініть SEED/LR.")

    for dec in (DECIMALS, 6):
        w = weights_numpy(model, dec)
        text, ended = decode_with(w, vocab)
        if ended and text == args.message:
            break
    print(f"\nЗберігаємо з точністю {dec} знаків. Декодинг: {text!r} (кінець ']' : {ended})")

    os.makedirs(args.out, exist_ok=True)
    for name, arr in w.items():
        with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(arr.tolist(), f)
    with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    def load(n):
        with open(os.path.join(args.out, f"{n}.json"), encoding="utf-8") as f:
            return np.array(json.load(f), dtype=np.float32)
    loaded = {n: load(n) for n in w}
    for n, a in loaded.items():
        print(f"  {n:18s} {a.shape}")
    text, ended = decode_with(loaded, vocab)
    print(f"\nПеревірка з файлів: {text!r}")
    print("OK" if ended and text == args.message else "ПОМИЛКА: файли не відтворюють повідомлення")


if __name__ == "__main__":
    main()