# Base image with Python
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential

# Copy project files
COPY . .

# Install uv + dependencies
RUN pip install uv
RUN uv sync --frozen  # uses pyproject.toml + uv.lock

# Expose port 8000
EXPOSE 8000

# Run the app
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"]
