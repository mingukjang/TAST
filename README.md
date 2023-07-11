### TAST ###
This code is for the paper "Test-time Adaptation via Self-training with Nearest Neighbor information (TAST)", accetped to ICLR'23.
We use the publicly released code "https://github.com/matsuolab/T3A".
You can follow the descriptions about installation and experiments in "https://github.com/matsuolab/T3A".

### Dependencies ###
Python 3.7.11
PyTorch 1.8.0
Torchvision 0.9.0
CUDA 10.2
CUDNN 7605
NumPy 1.2
PIL 8.4.0

### Data ###
You can download the domain generalization benchmarks, namely VLCS, PACS, OfficeHome, and TerraIncognita by following the procedure.
e.g.) python -m domainbed.scripts.download --data_dir=/my/datasets/path --dataset pacs

You can change pacs to vlcs, office_home, terra_incognita to download other datasets.

### Train ###
You can train a model on training domains.
e.g.) python -m domainbed.scripts.train\
       --data_dir /my/datasets/path\
       --output_dir /my/pretrain/path\
       --algorithm ERM\
       --dataset PACS\
       --hparams "{\"backbone\": \"resnet18-BN\"}" 

You can use backbone networks such as resnet50-BN, resnet50 which are presented in the train.py file.
The trained network and information about the training are recorded in "/my/pretrain/path"

### Test-time adaptation ###
While testing, we adapt trained classifiers.
e.g.)  python -m domainbed.scripts.unsupervised_adaptation\
       --input_dir=/my/pretrain/path\
       --adapt_algorithm=TAST

You can use the test-time adaptation algorithms such as T3A, TAST, and TAST_bn which are presented in the adapt_algorithms.py file.

Then, the test reulsts will be recorded in "/my/pretrain/path/out_TAST.txt" and "/my/pretrain/path/results_TAST.jsonl"
