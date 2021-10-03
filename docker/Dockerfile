FROM python:slim-buster

LABEL com.nvidia.volumes.needed=nvidia_driver

WORKDIR /forestock

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt