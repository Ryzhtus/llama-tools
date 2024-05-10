FROM nvidia/cuda:12.1.0-base-ubuntu22.04

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN mkdir /llm
COPY requirements.txt /llm
WORKDIR /llm
RUN pip install -r requirements.txt
COPY ./ llm
COPY ./src /llm/src
EXPOSE 8001
CMD ["python3", "-m", "uvicorn", "llm.llm:app", "--reload", "--host", "0.0.0.0", "--port", "8001"]