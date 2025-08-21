# Use a Python base image with GPU support
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with GPU support (matches your torch version)
RUN pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Copy the script
COPY erotic_story_generator.py .

# Set the command to run your script with RunPod handler
CMD ["python", "erotic_story_generator.py"]
