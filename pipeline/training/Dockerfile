FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    transformers==4.47.0 \
    peft==0.15.2 \
    trl==0.17.0 \
    "bitsandbytes>=0.46.1" \
    datasets==3.6.0 \
    accelerate==1.6.0 \
    boto3 \
    sentencepiece \
    protobuf \
    tiktoken \
    mlflow-skinny

COPY finetune_job.py /opt/scripts/finetune_job.py
COPY entrypoint.sh /opt/scripts/entrypoint.sh
RUN chmod +x /opt/scripts/entrypoint.sh

ENTRYPOINT ["/opt/scripts/entrypoint.sh"]
