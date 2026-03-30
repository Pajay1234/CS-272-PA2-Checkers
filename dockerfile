FROM python:3.12-slim

# Install system dependencies for Gymnasium/PettingZoo
RUN apt-get update && apt-get install -y \
    freeglut3-dev \
    swig \
    ffmpeg \
    libosmesa6-dev \
    python3-opengl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your local code into the container
COPY . .

# Set the default command to bash
CMD ["/bin/bash"]