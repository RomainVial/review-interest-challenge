# Priceminister Data Challenge

Code for [PriceMinister challenge](https://challengedata.ens.fr/en/challenge/26/prediction_of_products_reviews_interests.html) submission

## Dependencies

- Keras(2.0.3) with Theano(0.9.0) backend
- spacy(>=1.8.2) for tokenization and french word vectors
- scikit-learn(0.18.1) for ROC AUC metric

## Get the pretrained word vectors (1.4 GB)

```
sudo python -m spacy download fr
```

## Unzip the data

```
unzip data/data.zip -d data/
```

## Results

On the validation set, be sure to use the 821 seed.

### Using 300-dim spacy embedding

| Model                | Val   | Test  |
|----------------------|:-----:|:-----:|
| CBOW                 | 70.34 |   -   |
| LSTM 64              | 72.74 |   -   |
| biLSTM 32            | 72.52 |   -   |
| Average Ensemble     | 73.48 | 70.20 |

### Using 200-dim word2vec (old)

| Model         | Val   | Test  |
|---------------|:-----:|:-----:|
| LSTM 64       | 71.93 |   -   |
| biLSTM 32     | 71.93 |   -   |
| AttLSTM 64    | 71.84 |   -   |
| Att biLSTM 64 | 71.71 |   -   |
| Mean Ensemble | 72.86 |   -   |
| Best Ensemble | 72.88 | 70.31 |
