
# Copyright (c) Microsoft Corporation. All rights reserved.
# Dockerfile
# https://eng.ms/docs/more/containers-secure-supply-chain/approved-images
FROM  mcr.microsoft.com/oryx/python:3.10

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg
RUN apt-get update -y
ENV IS_DOCKER True

# Install dependencies
WORKDIR ./
COPY . .
RUN pip install -r requirements.txt

# Run application
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "./app/Home.py"]