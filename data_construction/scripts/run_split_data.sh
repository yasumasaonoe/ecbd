DATA_DIR='YOUR_DATA_DIR'

YEAR='2022'
python split_data.py \
    --read_from ${DATA_DIR}/masked_sentences_spans_noun_phrases_${YEAR}.json \
    --save_to ${DATA_DIR}/${YEAR}_np_spans \
    --definition_file ${DATA_DIR}/sentences_${YEAR}.json \
    --span_file ${DATA_DIR}/masked_sentences_spans_noun_phrases_${YEAR}.json \
    --entity_file_path ${DATA_DIR}/entity_splits_${YEAR}.json \
    --distance 10 \
    --min_n_sentences 1 \
    --max_n_sentences 100

python split_data.py \
    --read_from ${DATA_DIR}/masked_sentences_spans_random_${YEAR}.json \
    --save_to ${DATA_DIR}/${YEAR}_random_spans \
    --definition_file ${DATA_DIR}/sentences_${YEAR}.json \
    --span_file ${DATA_DIR}/masked_sentences_spans_random_${YEAR}.json \
    --entity_file_path ${DATA_DIR}/entity_splits_${YEAR}.json \
    --distance 10 \
    --min_n_sentences 1 \
    --max_n_sentences 100


YEAR='2023'
python split_data.py \
    --read_from ${DATA_DIR}/masked_sentences_spans_noun_phrases_${YEAR}.json \
    --save_to ${DATA_DIR}/${YEAR}_np_spans \
    --definition_file ${DATA_DIR}/sentences_${YEAR}.json \
    --span_file ${DATA_DIR}/masked_sentences_spans_noun_phrases_${YEAR}.json \
    --entity_file_path ${DATA_DIR}/entity_splits_${YEAR}.json \
    --distance 10 \
    --min_n_sentences 1 \
    --max_n_sentences 100

python split_data.py \
    --read_from ${DATA_DIR}/masked_sentences_spans_random_${YEAR}.json \
    --save_to ${DATA_DIR}/${YEAR}_random_spans \
    --definition_file ${DATA_DIR}/sentences_${YEAR}.json \
    --span_file ${DATA_DIR}/masked_sentences_spans_random_${YEAR}.json \
    --entity_file_path ${DATA_DIR}/entity_splits_${YEAR}.json \
    --distance 10 \
    --min_n_sentences 1 \
    --max_n_sentences 100