# Copyright (c) Microsoft Corporation. All rights reserved.
# https://eng.ms/docs/more/containers-secure-supply-chain/approved-images
FROM  mcr.microsoft.com/oryx/python:3.11

RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    apt-get update -y && \
    apt-get install wkhtmltopdf -y && \
    curl -sSL https://install.python-poetry.org | python -

ENV PATH="/root/.local/bin:$PATH"

COPY . .
RUN rm -rf .git .streamlit/app_secrets.toml .vscode/ .github/ .gitignore
RUN poetry install --only main

# Run application
EXPOSE 80
ENTRYPOINT ["poetry", "run", "poe", "run_streamlit" "--server.port=80"]