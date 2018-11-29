FROM python:3.6

RUN adduser --disabled-password codebreaker-mi

WORKDIR /home/codebreaker-mi

COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install --upgrade pip
RUN venv/bin/pip install -r requirements.txt
RUN venv/bin/pip install gunicorn

COPY app app
COPY model model
COPY results results
COPY application.py codebreaker_mi.py boot.sh ./
RUN chmod +x boot.sh

RUN chown -R codebreaker-mi:codebreaker-mi ./
USER codebreaker-mi

EXPOSE 8080
ENTRYPOINT ["./boot.sh"]