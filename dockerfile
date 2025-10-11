FROM python:3.11-slim

WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# create venv inside container
RUN python -m venv /app/.venv

# upgrade pip inside venv
RUN /app/.venv/bin/pip install --upgrade pip

# copy requirements and install inside venv
COPY requirements.txt .
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# default command uses container venv
ENTRYPOINT ["/app/.venv/bin/python"]
CMD ["main.py"]
