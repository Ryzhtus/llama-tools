FROM python:3.11
RUN mkdir /app
COPY requirements.txt /app
WORKDIR /app
RUN pip install -r requirements.txt
COPY ./ app
COPY ./src /app/src
CMD ["streamlit", "run", "app/app.py"]