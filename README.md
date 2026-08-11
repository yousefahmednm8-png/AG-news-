# AG News Classification with Transformer — PyTorch

A text classification project using a Transformer Encoder built with PyTorch to classify news articles into four categories:

**World · Sports · Business · Sci/Tech**

## Pipeline

* Text preprocessing & cleaning
* Tokenization
* Vocabulary building
* Numericalization
* Dynamic padding & attention masks
* Sinusoidal positional encoding
* Transformer Encoder
* Training & validation
* Model checkpointing
* Test evaluation
* Confusion matrix
* Inference with `predict.py`

## Model

```text
Embedding Dimension: 128
Attention Heads: 4
Transformer Layers: 4
Feed Forward Dimension: 128
Batch Size: 64
Dropout: 0.2
Optimizer: AdamW
Epochs: 8
```

## Results

**Test Accuracy: 90.29%**

## Inference

The trained model can be used to classify new articles:

```bash
python predict.py
```

Example:

```text
Enter news article: Apple announced a new technology product today

Prediction: Sci/Tech
Confidence: XX.XX%
```

## Project Structure

```text
AG-News-Transformer/
│
├── AG news.py
├── predict.py
├── best_transformer_model.pth
├── vocab.pkl
├── README.md
└── images/
    ├── training_curves.png
    └── confusion_matrix.png
```

## Technologies

Python · PyTorch · Pandas · Scikit-learn · Matplotlib

## What I Learned

This project helped me practice building a complete NLP classification pipeline with PyTorch, including vocabulary handling, attention masks, positional encoding, Transformer encoders, training, evaluation, and model inference.

## Trained Model

The trained model is available on Google Drive:

https://drive.google.com/drive/folders/1cmc8MeU2aAvMltBzjngOR3L0TC2P_txQ?usp=sharing

## Future Improvements

* Experiment with larger Transformer architectures
* Add learning-rate scheduling
* Explore pretrained language models
* Deploy the model as an API
