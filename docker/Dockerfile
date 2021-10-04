FROM python:slim-buster

LABEL com.nvidia.volumes.needed=nvidia_driver

RUN useradd -ms /bin/bash scientist

USER scientist

WORKDIR /home/scientist/forestock

ENV PATH="/home/scientist/.local/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt