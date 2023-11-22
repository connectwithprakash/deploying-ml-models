# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY . /app

# RUN git clone https://github.com/connectwithprakash/fetch-machine-learning-exercise.git .

RUN pip3 install -r requirements.txt

EXPOSE 8090

HEALTHCHECK CMD curl --fail http://localhost:8090/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8090", "--server.address=0.0.0.0"]
