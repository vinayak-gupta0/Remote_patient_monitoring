# Start with a lightweight Linux that has Python 3.9 installed
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all your files into the container
COPY . .

# Install your libraries
RUN pip install --no-cache-dir -r requirements.txt

# Tell the container how to start (using the $PORT variable provided by Tsuru)
CMD sh -c 'python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0'
