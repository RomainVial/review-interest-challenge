# Priceminister Data Challenge

Code for [PriceMinister challenge](https://challengedata.ens.fr/en/challenge/26/prediction_of_products_reviews_interests.html) submission

## Dependencies

- Keras(2.0.3) with Theano(0.9.0) backend
- spacy(1.8.0) for tokenization

## Get the pretrained word vectors

´´´
cd data/
wget http://embeddings.org/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin
git clone https://github.com/marekrei/convertvec
cd convertvec/
make
cd ..
convertvec/convertvec bin2txt frWiki_no_phrase_no_postag_700_cbow_cut100.bin word2vec_fr_200
´´´

## Results

On the validation set, be sure to use the 821 seed.

| Model       | Val   | Test  |
|-------------|:-----:|:-----:|
| LSTM 64     | 70.57 | 68.62 |
| biLSTM 32   | 70.95 |   -   |
| AttLSTM 32  | 70.97 |   -   |
| AttLSTM 64  | 71.23 |   -   |
| Avg Ensemble| 72.17 | 69.74 |