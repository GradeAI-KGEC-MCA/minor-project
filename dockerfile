# base image
FROM python:3.11-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# create a venv outside /app so mounts won't overwrite it
RUN python -m venv /venv

# upgrade pip inside venv
RUN /venv/bin/pip install --upgrade pip

# copy requirements and install inside venv
COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# entrypoint points to container venv python
ENTRYPOINT ["/venv/bin/python"]
CMD ["main.py"]
