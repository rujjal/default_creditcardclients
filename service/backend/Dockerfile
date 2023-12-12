FROM python:3.10-bookworm

#FROM  spark:3.5.0-scala2.12-java17-ubuntu

# change the user for executing operations
USER root

# updating apt packet manager
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless
RUN  apt-get clean

# updating pip
RUN pip install --no-cache-dir --upgrade pip

RUN pip install pyspark

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

# starting the app
CMD ["python3", "-m", "api"]