FROM python:3.10

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=80
ENV GRADIO_DEBUG=1

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN pip install git+https://github.com/andreaschandra/TinyTroupeOllama.git@main --no-dependencies

CMD ["python", "app.py"]