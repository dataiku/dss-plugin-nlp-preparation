ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /plugin

# Copy requirements files first (for better layer caching)
COPY code-env/python/spec/requirements.txt /plugin/requirements-plugin.txt
COPY tests/python/unit/requirements.txt /plugin/requirements-test.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-test.txt && \
    pip install --no-cache-dir -r requirements-plugin.txt

# Copy source code and resources
COPY python-lib/ /plugin/python-lib/
COPY resource/ /plugin/resource/
COPY tests/python/unit/ /plugin/tests/python/unit/

# Set environment variables
ENV PYTHONPATH=/plugin/python-lib
ENV DICTIONARY_FOLDER_PATH=/plugin/resource/dictionaries
ENV STOPWORDS_FOLDER_PATH=/plugin/resource/stopwords

# Run tests with system info
CMD echo "=== System Info ===" && \
    echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2)" && \
    echo "Architecture: $(uname -m)" && \
    echo "Python: $(python --version)" && \
    echo "Pip: $(pip --version)" && \
    echo "==================" && \
    pytest tests/python/unit -v
