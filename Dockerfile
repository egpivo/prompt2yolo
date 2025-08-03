# Author: Joseph Wang
# Version: 0.2.1

FROM python:3.10-slim

# Metadata
LABEL maintainer="Joseph Wang <joseph.wang@instai.co>" \
      version="0.2.1" \
      description="A pipeline for automatically generating training data using prompts and fine-tuning YOLO models."

# Set environment variables
ENV POETRY_HOME="/root/.local"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s $(python3 -c "import site; print(site.USER_BASE)")/bin/poetry /usr/local/bin/poetry

# Verify Poetry Installation
RUN poetry --version

# Install dependencies
COPY pyproject.toml poetry.lock README.md .env ./
COPY scripts/ scripts/
COPY envs/ envs/
COPY prompt2yolo prompt2yolo/

RUN poetry install --without dev
ENV PYTHONPATH="/app"

# Expose necessary ports (if running a Streamlit app, for example)
EXPOSE 8501

# Default command
CMD ["poetry", "run", "streamlit", "run", "--server.headless=true", "/app/prompt2yolo/experiment_manager/app.py"]
