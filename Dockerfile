# Copyright (c) Microsoft Corporation. All rights reserved.
# https://eng.ms/docs/more/containers-secure-supply-chain/approved-images
FROM  mcr.microsoft.com/oryx/python:3.11

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg
RUN apt-get update -y
RUN apt-get install wkhtmltopdf -y

# Install dependencies
WORKDIR ./
COPY ./app ./app
COPY ./python ./python
COPY ./README.md ./
COPY ./.streamlit ./.streamlit
COPY Dockerfile .
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run application
EXPOSE 8501
ENTRYPOINT ["python", "-m", "streamlit", "run", "app/Home.py"]