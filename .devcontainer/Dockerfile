FROM mcr.microsoft.com/devcontainers/python:0-3.7

# RUN apt-get update && apt-get install libgl1 -y

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip && pip3 install -r requirements.txt
# RUN pip install -r requirements.txt