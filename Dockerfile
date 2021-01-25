FROM python:3.6.1-slim

RUN  mkdir -p /project && pip install --upgrade pip

WORKDIR /project

COPY requirements.txt /project

RUN pip install --no-cache-dir -r requirements.txt

COPY . /project

ENTRYPOINT [ "python" ]

CMD [ "run.py" ]