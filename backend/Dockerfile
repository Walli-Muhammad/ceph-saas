FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install wget and download model weights from Github Release
RUN apt-get update && apt-get install -y wget
RUN mkdir -p /app/weights

# Replace the fake URLs below with your real copied links!
RUN wget -O /app/weights/best_model.pth "sha256:625e3b77252ff44912437e1b69e45ae1a5f8f9e45690b322f2db3f7475b0c7de"
RUN wget -O /app/weights/final_model.pth "sha256:d14677fb202706012529ddbfdba605ef21cc9b75e2081bcd35d4e8b398e533ea"

# Copy the rest of the backend files
COPY . .

# Expose the API port
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]