#!/usr/bin/env bash
# Downloads everything the AI service needs (all free / open source). Run once from the repo root.
set -euo pipefail
mkdir -p models/tessdata
echo ">> Tesseract Arabic + English OCR models"
curl -sSL -o models/tessdata/ara.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/ara.traineddata
curl -sSL -o models/tessdata/eng.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata
curl -sSL -o models/tessdata/osd.traineddata https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/osd.traineddata
echo ">> spaCy English model (used by Presidio)"
python -m spacy download en_core_web_sm
echo "done. Optional Arabic NER: pip install transformers torch && export LEGACYLENS_ARABIC_NER=1"
