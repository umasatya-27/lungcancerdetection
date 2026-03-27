# Use a Node base image
FROM node:18-slim

# Install Python + venv + system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \ 
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files and install Node dependencies
COPY package*.json ./
RUN npm install

# Copy Python requirements
COPY requirements.txt ./

# Create virtual environment
RUN python3 -m venv /opt/venv

# Activate venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Build frontend
RUN npm run build

# Expose port
EXPOSE 3000

# Start app
CMD ["npm", "start"]