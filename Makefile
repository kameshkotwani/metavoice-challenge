list-s3:
	aws s3 ls $(S3_BUCKET)/data/ --endpoint-url $(ENDPOINT_URL) --profile $(PROFILE) 

download-data:
	aws s3 cp $(S3_BUCKET)/data/ ../data/raw/ --recursive --endpoint-url $(ENDPOINT_URL) --profile $(PROFILE)

list-files:
	ls data/wav48_silence_trimmed/$(DIR)  -1 | wc -l

run-fastapi-server:
	uvicorn fastapi_server.app:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 120
