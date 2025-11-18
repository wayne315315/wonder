FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /wonder

# Install
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir tensorflow==2.17.1, tqdm==4.67.1

# Copy your local code into the container
COPY . .

# Run
CMD ["bash"]
