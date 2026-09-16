# 1. Start with a lightweight version of Python
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code and model files into the container
COPY main.py .
COPY model.pkl .
COPY model_columns.pkl .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]