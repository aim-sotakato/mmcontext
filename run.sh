clearml-task --project screening_mmcontext \
             --name screening_classification_sample \
             --repo https://github.com/aim-sotakato/mmcontext.git \
             --branch main \
             --script train.py \
             --args "dataset_id=bdbaccd54ac7414f886a7766e703e0fc dataset=aim gpu=0 loss=LA out=result_la cv=5 fold=0" \
             --queue a100x1a \
             --docker harbor.dev.ai-ms.com/screening_mmcontext/mmcontext_docker_image:latest
