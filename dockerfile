# Start with the OpenFOAM base image
FROM microfluidica/openfoam:latest

# Install Python 3.12
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.13 python3.13-venv python3.13-dev python3-pip

# Set Python 3.12 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1
RUN update-alternatives --set python3 /usr/bin/python3.13

COPY Code /app

WORKDIR /app

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt
