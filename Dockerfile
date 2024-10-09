# Copyright (c) Microsoft Corporation. All rights reserved.
# https://eng.ms/docs/more/containers-secure-supply-chain/approved-images
FROM  mcr.microsoft.com/oryx/python:3.11 AS base

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    apt update -y && \
    apt install wkhtmltopdf -y && \
    apt-get install wkhtmltopdf locales -y && \
    curl -sSL https://install.python-poetry.org | python -

RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf && \
    locale-gen en_US.UTF-8

ENV PATH="/root/.local/bin:$PATH"

FROM base AS streamlit

COPY . .
RUN rm -rf .git .streamlit/app_secrets.toml .vscode/ .github/ .gitignore
RUN poetry install --only main

# Run application
EXPOSE 80
ENTRYPOINT ["poetry", "run", "poe", "run_streamlit", "--server.port=80"]