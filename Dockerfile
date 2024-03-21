# Use an official lightweight Python image.
FROM python:3.11-slim

# Set the working directory in the Docker container to /app
WORKDIR /app

# Copy the Poetry configuration files into the container
COPY pyproject.toml poetry.lock* ./

# Install system dependencies required for certain Python packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry, configure it to not create a virtual environment
# and install the project dependencies
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-interaction --no-ansi


# Copy the rest of your app's source code from your host to your image filesystem.
# Assuming your source code is in the src/ directory in your project
COPY src/ ./src

# Change the working directory to /app/src where your app.py and other source files are
WORKDIR /app/src

# Expose the port your app runs on
EXPOSE 8501

# Define the command to run your app using Poetry
CMD ["poetry", "run", "streamlit", "run", "src/app/main.py"]
