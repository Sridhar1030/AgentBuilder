FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    peft==0.15.2 \
    trl==0.17.0 \
    "bitsandbytes>=0.46.1" \
    datasets==3.6.0 \
    accelerate==1.6.0 \
    boto3 \
    sentencepiece \
    protobuf

COPY finetune_job.py /opt/scripts/finetune_job.py

ENTRYPOINT ["python", "/opt/scripts/finetune_job.py"]
