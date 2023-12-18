# the same image used for the implementation of the notebook
# I used this image to avoid spark version mismatch between 
# the version used for creating the model, and the one used to load it in the backend
FROM quay.io/jupyter/pyspark-notebook

# change the user for executing operations
USER root

# updating apt packet manager
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless
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

# starting the app
CMD ["python3", "-m", "api"]