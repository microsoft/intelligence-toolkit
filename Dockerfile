# Copyright (c) Microsoft Corporation. All rights reserved.
# https://eng.ms/docs/more/containers-secure-supply-chain/approved-images

# Use multi-stage builds
FROM mcr.microsoft.com/oryx/python:3.11 AS builder

# Install dependencies
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    apt-get update -y && \
    apt-get install wkhtmltopdf -y && \
    curl -sSL https://install.python-poetry.org | python - && \
    poetry install --no-dev

# Copy application files
COPY . .

# Final stage
FROM mcr.microsoft.com/oryx/python:3.11

# Copy only necessary files from the builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/share/keyrings/microsoft-prod.gpg /usr/share/keyrings/microsoft-prod.gpg
COPY --from=builder /app /app
COPY --from=builder /toolkit /toolkit
COPY --from=builder /example_outputs /example_outputs

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"

# Expose port and set entrypoint
EXPOSE 8501
ENTRYPOINT ["poetry", "run", "poe", "run_streamlit"]