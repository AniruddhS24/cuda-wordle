# Start from the Ubuntu 20.04 LTS image
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Update package list and install g++
RUN apt-get update && apt-get install -y g++

# Install Python 3.8 and pip
RUN apt-get install -y python3-pip python3-venv


# Verify Python version and pip installation
RUN python3 --version && pip --version

# Set working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install -r requirements.txt

RUN make build_interactive

# Set entrypoint to bash
# ENTRYPOINT ["/bin/bash"]

CMD ["python3", "app.py"]
