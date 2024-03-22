FROM python:3.9
EXPOSE 8080
COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt
COPY . .
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "mnist_inference.py", "--server.port=8080", "--server.address=0.0.0.0"]