DATA_DIR='YOUR_DATA_DIR'

# 2022
YEAR='2022'
python -u select_wikipedia_sentences.py \
    --enwiki_dir ${DATA_DIR}/enwiki_20230401 \
    --target_pageids ${DATA_DIR}/${YEAR}_entities.txt \
    --cached_wiki_pages ${DATA_DIR}/entities_with_metadata_${YEAR}.json \
    --save_to ${DATA_DIR}/sentences_${YEAR}.json

# 2023
YEAR='2023'
python -u select_wikipedia_sentences.py \
    --enwiki_dir ${DATA_DIR}/enwiki_20230401 \
    --target_pageids ${DATA_DIR}/${YEAR}_entities.txt \
    --cached_wiki_pages ${DATA_DIR}/entities_with_metadata_${YEAR}.json \
    --save_to ${DATA_DIR}/sentences_${YEAR}.json

