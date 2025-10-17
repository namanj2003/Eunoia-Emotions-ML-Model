FROM python:3.10-slim

WORKDIR /app

# Install dependencies directly
RUN pip install --no-cache-dir flask==3.0.0 flask-cors==4.0.0 requests==2.31.0

# Copy application code
COPY app_final.py app.py

# Environment
ENV PORT=7860
EXPOSE 7860

# Run
CMD ["python", "app.py"]
