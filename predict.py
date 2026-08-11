import re
import pickle
import torch
import torch.nn as nn


# =========================
# Configuration
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "best_transformer_model.pth"
VOCAB_PATH = "vocab.pkl"

CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Sci/Tech"
]


# =========================
# Load Vocabulary
# =========================

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)


# =========================
# Text Preprocessing
# =========================

def clean_text(text):
    text = text.lower()
    text = text.replace("\\", " ")
    text = re.sub(r"#\d+;", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9.,!?'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    return text.split()


def numericalize(tokens):
    return [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens
    ]


# =========================
# Positional Encoding
# =========================

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)



class AGNewsTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        num_classes,
        pad_idx,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.positional_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )

    def forward(self, x, attention_mask):

        x = self.embedding(x)

        x = self.positional_encoding(x)

        padding_mask = ~attention_mask

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        mask = attention_mask.unsqueeze(-1)

        x = (x * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1)

        x = self.classifier(x)

        return x


# =========================
# Create Model
# =========================

model = AGNewsTransformer(
    vocab_size=len(vocab),
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    ff_dim=128,
    num_classes=4,
    pad_idx=vocab["<PAD>"],
    dropout=0.2
).to(DEVICE)


# =========================
# Load Trained Weights
# =========================

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model.eval()


# =========================
# Prediction Function
# =========================

def predict(text):

    text = clean_text(text)

    tokens = tokenize(text)

    ids = numericalize(tokens)

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    attention_mask = (
        x != vocab["<PAD>"]
    )

    with torch.no_grad():

        outputs = model(
            x,
            attention_mask
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        prediction = outputs.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0, prediction
        ].item()

    return (
        CLASS_NAMES[prediction],
        confidence
    )


# =========================
# Interactive Prediction
# =========================

print("AG News Classifier")
print("Type 'exit' to quit.\n")

while True:

    text = input("Enter news article: ")

    if text.lower() == "exit":
        break

    category, confidence = predict(text)

    print(f"\nPrediction: {category}")
    print(f"Confidence: {confidence * 100:.2f}%\n")
