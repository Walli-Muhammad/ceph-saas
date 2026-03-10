FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install wget and download model weights from Github Release
RUN apt-get update && apt-get install -y wget
RUN mkdir -p /app/checkpoints

# Replace the fake URLs below with your real copied links!
RUN wget -O /app/checkpoints/best_model.pth "https://github.com/Walli-Muhammad/ceph-saas/releases/download/v1.0/best_model.pth"
RUN wget -O /app/checkpoints/final_model.pth "https://github.com/Walli-Muhammad/ceph-saas/releases/download/v1.0/final_model.pth"

# Copy the rest of the backend files
COPY . .

# Expose the API port
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]