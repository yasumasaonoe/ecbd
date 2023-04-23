YEAR='2023'
DATA_DIR='YOUR_DATA_DIR'

python mask_sentences.py \
    --read_from ${DATA_DIR}/sentences_${YEAR}.json \
    --save_to ${DATA_DIR}/masked_sentences_spans_noun_phrases_${YEAR}.json\
    --masking_mode span \
    --mask_type noun_phrases

python mask_sentences.py \
    --read_from ${DATA_DIR}/sentences_${YEAR}.json \
    --save_to ${DATA_DIR}/masked_sentences_spans_random_${YEAR}.json\
    --masking_mode span \
    --mask_type random
