FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN python3 -m nltk.downloader stopwords
RUN python3 -m spacy download ru_core_news_sm

COPY ./models /root/nltk_data
COPY ./models ./models
COPY run.py .

CMD python3 run.py --input-path $INPUT_PATH --output-path $OUTPUT_PATH
