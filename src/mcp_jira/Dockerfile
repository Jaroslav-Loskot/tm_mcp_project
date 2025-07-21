FROM python:3.12-slim

WORKDIR /app

COPY ../mcp-common /mcp-common
COPY . /app

RUN pip install --upgrade pip \
 && pip install "mcp-common @ file:///mcp-common" \
 && pip install .

CMD ["python", "main.py"]