# metavoice-challenge

Data Engineering challenge by metavoice

## Steps performed

    1. 

1. Download data from s3 using aws cli:

    ```bash
    aws s3 cp $(S3_BUCKET)/data/ ../data/raw/ --recursive --endpoint-url $(ENDPOINT_URL) --profile $(PROFILE)
    ```

### Tech Used

1. Whisper: https://github.com/openai/whisper
2. Mock Code: https://gist.github.com/sidroopdaska/364e9f493d8dd9584eb9e1e9cae5715c
3. Airflow for the data pipeline
4. FastAPI server for transcriptions
