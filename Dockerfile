# Use a slim Python image
FROM python:3.12-slim

# Set working directory (optional, good for debugging into the container)
WORKDIR /app

# Copy all source code into /src
COPY src /src

# Install curl and system dependencies (if needed), upgrade pip and install Python packages
RUN apt-get update && apt-get install -y curl \
 && pip install --upgrade pip \
 && pip uninstall -y fastmcp || true \
 && pip install fastmcp==2.10.6 \
 && pip install /src/mcp_jira \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Ensure Python can find /src for module imports like `mcp_jira.main`
ENV PYTHONPATH="/src"

# Expose FastAPI app port
EXPOSE 8001

# ✅ FIX: Run using Python module style to support relative imports and match local dev
CMD ["python", "-m", "mcp_jira.main"]
