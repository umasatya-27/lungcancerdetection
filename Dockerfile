cat <<EOF > Dockerfile
FROM node:20-bookworm
RUN apt-get update && apt-get install -y python3 python3-pip python3-dev build-essential libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV NODE_ENV=production
COPY package.json ./
RUN rm -rf node_modules package-lock.json && npm install
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --break-system-packages --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
EOF