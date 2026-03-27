# Use a stable Node image
FROM node:20-slim

# Install system dependencies for OpenCV and Python
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

# Set production environment
ENV NODE_ENV=production

# Copy package files and install Node dependencies
COPY package.json ./
RUN npm install

# Install Python dependencies ONE BY ONE to save RAM
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