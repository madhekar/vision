# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye
#FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_1.txt .

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# RUN git -b my_branch clone https://github.com/madhekar/vision.git .
# research/code/test/multipage_app
# RUN git clone https://github.com/streamlit/streamlit-example.git .

# RUN git clone --single-branch --branch /research/code/test/multipage_app  https://github.com/madhekar/vision.git
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir  --default-timeout=900 -r requirements_1.txt

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]