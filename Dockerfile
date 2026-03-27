# Use a more robust Node image
FROM node:20-bookworm

# Install Python and system dependencies for OpenCV and native builds
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy ONLY package.json first
COPY package.json ./

# NUCLEAR FIX: Delete any potential lock files and force a fresh install for Linux
RUN rm -rf node_modules package-lock.json && npm install

# Copy Python requirements and install Python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --break-system-packages --extra-index-url https://download.pytorch.org/whl/cpu torch
RUN pip3 install --no-cache-dir --break-system-packages --extra-index-url https://download.pytorch.org/whl/cpu torchvision
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the rest of the application
COPY . .

# Build the frontend
RUN npm run build

# Expose the port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]