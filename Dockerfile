FROM python:3.9

ENV PYTHONPATH=/app

RUN pip install tensorflow-cpu numpy gradio

ADD requirements.txt app.py /app/
ADD models/context_encoder*.h5 /app/models/
ADD context_encoder/common.py /app/context_encoder/common.py

WORKDIR /app

EXPOSE 7860

CMD ["python", "app.py"]
