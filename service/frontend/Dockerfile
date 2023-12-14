FROM python:3.10-bookworm

# you can use basic python image for the frontend to make it more light
# FROM spark:3.5.0-scala2.12-java17-python3-ubuntu

# change the user for executing operations
USER root

# updating apt packet manager
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless
RUN  apt-get clean

# updating pip
RUN pip install --no-cache-dir --upgrade pip

# you dont need it in the frontend
# RUN pip install pyspark
# RUN pip install mlflow

# TIPS:
# if you want to install some python libraries in a container
# if they are more than one it's recommended to use a requirements.txt file 
# and insert inside all the neede library and let pip install it (command in line 32)

# moving to the working directory in the container
WORKDIR /home

# copying in the working directory the requirements file
COPY requirements.txt .

# installing python requirements
RUN pip install -r requirements.txt

# creating directory structure in the current working directory in the container
RUN mkdir -p src

# copying src file from local (1) directory to container directory (2)
COPY ./templates ./src/templates
COPY ./app.py ./src

# switching working directory
WORKDIR /home/src

# starting the app
CMD ["python3", "-m", "app"]