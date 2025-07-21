# Use a slim Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all source code into /src
COPY src /src

# Install curl and system dependencies, then upgrade pip and install Python packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gcc libffi-dev libssl-dev \
 && pip install --upgrade pip \
 && pip install /src/mcp_common \
 && pip install /src/mcp_jira \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Ensure Python can find /src for imports
ENV PYTHONPATH="/src"

# Expose FastAPI app port
EXPOSE 8001

# Run the main FastMCP server
CMD ["python", "/src/mcp_jira/main.py"]
