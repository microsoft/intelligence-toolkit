# Copyright (c) Microsoft Corporation. All rights reserved.
# https://eng.ms/docs/more/containers-secure-supply-chain/approved-images
FROM  mcr.microsoft.com/oryx/python:3.11

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg
RUN apt-get update -y
RUN apt-get install wkhtmltopdf -y
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"

# Install dependencies
WORKDIR ./
COPY Dockerfile .
COPY pyproject.toml .
COPY poetry.lock .
COPY ./.streamlit ./.streamlit
COPY ./README.md ./

COPY ./app ./app
COPY ./toolkit ./toolkit

RUN poetry install

# Run application
EXPOSE 8501
ENTRYPOINT ["poetry", "run", "poe", "run_streamlit"]