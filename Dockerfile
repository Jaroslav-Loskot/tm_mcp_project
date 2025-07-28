FROM python:3.12-slim

WORKDIR /app

# Copy all sources under /src
COPY src /src

RUN apt-get update && apt-get install -y curl \
 && pip install --upgrade pip \
 && pip install -r /src/requirements.txt \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


# Tell Python where to find your packages
ENV PYTHONPATH="/src"

# Expose FastAPI port
EXPOSE 8100

# Run the main script directly
CMD ["python", "/src/mcp_jira/main.py"]
