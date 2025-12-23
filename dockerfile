FROM node:20-bookworm

# Python
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything
COPY . .

# Install python package + deps (pyproject.toml exists)
RUN pip3 install --no-cache-dir -e .

# Install node deps
WORKDIR /app/web
RUN npm ci || npm install

ENV PORT=3000
EXPOSE 3000

CMD ["node", "server.js"]
