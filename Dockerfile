FROM python:3.12-slim

# Set working dir to the repo root
WORKDIR /app

# Copy entire project structure (src/ will be under /app/src)
COPY ../.. /app

# Set PYTHONPATH so both mcp_jira and mcp_common are importable
ENV PYTHONPATH=/app/src

# Install dependencies
RUN pip install --upgrade pip \
 && pip install "mcp-common @ file:///app/src/mcp_common" \
 && pip install .

# Move to src so we can run modules properly
WORKDIR /app/src

# Run the Jira module
CMD ["python", "-m", "mcp_jira.main"]
