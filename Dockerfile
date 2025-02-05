# Copyright (c) Microsoft Corporation. All rights reserved.
# https://eng.ms/docs/more/containers-secure-supply-chain/approved-images
FROM  mcr.microsoft.com/oryx/python:3.11
LABEL org.opencontainers.image.source=https://github.com/microsoft/intelligence-toolkit

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    apt update -y && \
    apt install wkhtmltopdf -y && \
    apt-get install wkhtmltopdf -y && \
    curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.3 python -
        
ENV PATH="/root/.local/bin:$PATH"

COPY . .
RUN poetry install --only main

# Run application
EXPOSE 80
ENTRYPOINT ["poetry", "run", "poe", "run_streamlit", "--server.port=80", "--server.address=0.0.0.0"]
