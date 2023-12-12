FROM python:3.10-bookworm

FROM spark:latest

#ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

# change the user for executing operations
USER root

# updating apt packet manager
RUN apt-get update && apt-get install -y --no-install-recommends 
RUN  apt-get clean

# updating pip
RUN pip install --no-cache-dir --upgrade pip

# moving to the working directory in the container
WORKDIR /home

# copying in the working directory the requirements file
COPY requirements.txt .

# installing python requirements
RUN pip install -r requirements.txt

# creating directory structure in the current working directory in the container
RUN mkdir -p src 
RUN mkdir -p /src/model

# copying src file from local (1) directory to container directory (2)
COPY ./model ./src/model
COPY ./api.py ./src

# switching working directory
WORKDIR /home/src

RUN apt-get update && \
    apt-get install -y python3

# starting the app
CMD ["python", "-m", "api"]