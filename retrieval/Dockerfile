FROM tiangolo/uvicorn-gunicorn:python3.11
RUN mkdir /retrieval
COPY requirements.txt /retrieval
WORKDIR /retrieval
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
COPY ./ retrieval
COPY ./src /retrieval/src
EXPOSE 8002
CMD ["uvicorn", "retrieval.retrieval:app", "--reload", "--host", "0.0.0.0", "--port", "8002"]