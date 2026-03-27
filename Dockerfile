# Use Node base image
FROM node:18-slim

# Install Python + venv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node dependencies
COPY package*.json ./
RUN npm install

# Install Python dependencies
COPY requirements.txt ./
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# 🔥 BUILD TYPESCRIPT (IMPORTANT)
RUN npm run build:server

# Expose port
EXPOSE 3000

# Start server (compiled JS)
CMD ["node", "dist/server.js"]