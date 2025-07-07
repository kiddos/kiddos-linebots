# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install git to handle git+ dependencies
RUN apt-get update && apt-get install -y git

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Expose the port the app runs on
EXPOSE 8001

RUN sed -i "1 i\__import__('pysqlite3')\nimport sys\nsys.modules['sqlite3'] = sys.modules.pop('pysqlite3')" main.py
ENV OLLAMA_HOST="http://host.docker.internal:11434"
ENV MONGO_HOST="host.docker.internal"
ENV CHROMA_HOST="host.docker.internal"
ENV CHROMA_PORT="8001"

# Run the application
CMD ["./prod.sh"]
