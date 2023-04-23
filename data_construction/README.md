# Data Construction

## Requirements
### English Wikipedia
This code requires English Wikipedia dump that can be downloaded from [here](https://dumps.wikimedia.org/enwiki/). To preprocess the raw dump, we use [WikiExtractor](https://github.com/attardi/wikiextractor), which extracts clean text from the dump. (In our paper, we used September 2021 dump. You can download the preprocessed dump (output from WikiExtractor) from [here](https://utexas.box.com/s/wxx9md1ejf8gcacbyvkedyznkf4y5o38).)  

## Example: Creating the 2022 and 2023 subsets.

### Step 0: Download EN Wiki dump and run WikiExtractor on it

```
wget https://dumps.wikimedia.org/enwiki/20230401/enwiki-20230401-pages-articles.xml.bz2
```

```
wikiextractor enwiki-20230401-pages-articles.xml.bz2 -o enwiki_20230401 --json -l
```
Preprocessed EN Wikipedia pages are saved in `./enwiki_20230401` as JSON lines.

### Step 1: Entity Mining

```
PROPS="P580 P9448 P6949 P575 P585 P571"
START_DATE=$(date -I)
END_DATE="2021-12-31"
OUTPUT_PATH="${START_DATE}_${END_DATE}.tsv"

DATE=$START_DATE
while [ "$DATE" != $END_DATE ]; do 
  for PROP in $PROPS; do
    echo $DATE $PROP
    wget --quiet -O - "https://query.wikidata.org/sparql?query=SELECT%20%3Fitem%20%3FitemLabel%20%0AWHERE%0A%7B%0A%20%20%3Fitem%20%20p%3A${PROP}/ps%3A${PROP}%20%22${DATE}T00%3A00%3A00Z%22%5E%5Exsd%3AdateTime.%0A%20%20%3Fitem%20rdfs%3Alabel%20%3FitemLabel.%0A%20%20FILTER%28LANG%28%3FitemLabel%29%20%3D%20%22en%22%29.%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22%5BAUTO_LANGUAGE%5D%2Cen%22.%20%7D%0A%7D" \
      | sed -r -n -e '/^\s+<uri>/p' -e '/^\s+<literal xml:lang=/p' \
      | sed -r 's/^\s+<.*>(.*)<\/.*>/\1/g' \
      | sed -r 's/^http:.*(Q[0-9]+)/\1/g' \
      | sed -r 'N;s/\n/\t/g' \
      | sed -r -e "s/$/\t${DATE}\t${PROP}/g" >> ${OUTPUT_PATH}
  done
  DATE=$(date -I -d "$DATE - 1 day")
done

```

```
PROPS="P580 P9448 P6949 P575 P585 P571"
ENTS_PATH="2023-03-29_2021-12-31.tsv"
OUTPUT_PATH='out.tsv'

while IFS=$'\t' read -r -a ENTITY; do
  for PROP in $PROPS; do
    echo "${ENTITY[0]} ${PROP}"
    wget --quiet -O - "https://query.wikidata.org/sparql?query=SELECT%20?item%20WHERE%20{%20wd:${ENTITY[0]}%20wdt:${PROP}%20?item.%20SERVICE%20wikibase:label%20{%20bd:serviceParam%20wikibase:language%20%22en,en%22.%20}%20}"  \
      | sed -r -n -e '/^\s+<literal datatype/p' \
      | sed -r 's/^\s+<.*>(.*)<\/.*>/\1/g' \
      | sed -r -e "s/$/\t${ENTITY[0]}\t${PROP}/g" >> ${OUTPUT_PATH}
  done
done < $ENTS_PATH
```

TODO: Add metadata. 

### Step 1: Select sentences from Wiki dump 
- Run `scripts/run_select_wikipedia_sentences.sh`

### Step 2: Mask entity spans 
- Run `scripts/run_mask_sentences.sh`

### Step 3: Get definition sentences 
- Run `scripts/run_select_wikipedia_definition_sentences.sh`

### Step 4: Split examples
- Run `scripts/run_split_data.sh`
