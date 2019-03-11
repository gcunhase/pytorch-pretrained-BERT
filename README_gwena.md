### Setup
```
pip install -r requirements.txt
```

* Installing [APEX](https://www.github.com/nvidia/apex) (needs CUDA 9.0+) DID NOT WORK (`No module named 'fused_layer_norm_cuda'`)
```
git clone https://github.com/NVIDIA/apex.git
cd apex && python setup.py install
```

### Fine-tuning with Microsoft Research Paraphrase Corpus (MRPC) Dataset
```
CUDA_VISIBLE_DEVICES=1,2 python run_classifier.py --task_name MRPC --do_train --do_eval --do_lower_case --data_dir /mnt/Gwena/pytorch-pretrained-BERT/glue_data/MRPC/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir /tmp/mrpc_output/
```
> Eval accuracy = 84.56%, Eval loss = 0.3755, global_step = 345, loss = 0.2343

### Fine-tuning with The Stanford Sentiment Treebank (SST) Dataset
```
CUDA_VISIBLE_DEVICES=1,2 python run_classifier.py --task_name SST --do_train --do_eval --do_lower_case --data_dir /mnt/Gwena/pytorch-pretrained-BERT/glue_data/SST-2/ --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir /tmp/sst_output/
```
> Modified `run_classifier.py` to accept SST classification task

>  Eval accuracy = 93.00%, Eval loss = 0.2476, global_step = 6315, loss = 0.06669
