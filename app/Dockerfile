FROM python:3.11
RUN mkdir /app
COPY requirements.txt /app
WORKDIR /app
RUN pip install -r requirements.txt
COPY ./ app
COPY ./src /app/src
EXPOSE 8000
CMD ["python", "-m", "streamlit", "run", "app/app.py", "--server.address", "0.0.0.0", "--server.port", "8000"]