FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install "fastapi[standard]"


COPY ./main.py /code/
COPY ./best_model.weights.h5 /code/

EXPOSE 80

CMD ["fastapi", "run", "main.py", "--port", "80"]
