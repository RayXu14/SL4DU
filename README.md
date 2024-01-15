# SL4DU

## Environment
### Option 1: container
frontlibrary/transformers-pytorch-gpu:4.6.1-pyarrow

```bash
~$ docker run --runtime=nvidia -it --rm -v $HOME/SL4DU:/workspace frontlibrary/transformers-pytorch-gpu:4.6.1-pyarrow
```

### Option 2: build from scatch
* Python==3.9 ([There may be some problem for numpy with 3.10](https://exerror.com/building-wheel-for-numpy-pyproject-toml/))
* nltk
* numpy (can be automatically installed when installing scipy)
* scipy (if you have problem, see this [solution](https://stackoverflow.com/questions/11114225/installing-scipy-and-numpy-using-pip))
* torch==1.8
* pyarrow
* tqdm
* transformers==4.5.1
* sklearn
* stop_words

## Reproduce step: an example
1. Initialize directories
    ```
    SL4DU
        code
        data
        pretrained
    ```
2. Download code and the Ubuntu data
    ``` bash
    ~/SL4DU/code$ git clone https://github.com/RayXu14/SL4DU.git
    ~/SL4DU/data$ wget https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
    ~/SL4DU/data$ unzip ubuntu_data.zip
    ```
3. Add [bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main) pretrained model in *pretrained*
    * config.json
    * vocab.txt
    * pytorch_model.bin
3. Preprocess data
    ```bash
    ~/SL4DU/code/SL4DU$ python3 preprocess.py --task=RS --dataset=Ubuntu --raw_data_path=../../data/ubuntu_data --pkl_data_path=../../data/ubuntu_data --pretrained_model=bert-base-uncased
    ```
4. Reproduce BERT result
    ```bash
    ~/SL4DU/code/SL4DU$ python3 -u train.py --save_ckpt --task=RS --dataset=Ubuntu --pkl_data_path=../../data/ubuntu_data --pretrained_model=bert-base-uncased --add_EOT --freeze_layers=0 --train_batch_size=8 --eval_batch_size=100 --log_dir=? # --pkl_valid_file=test.pkl
    ```
5. Add *post-ubuntu-bert-base-uncased* in *pretrained*
    * Download [whang's Ubuntu ckpt](https://drive.google.com/file/d/1jt0RhVT9y2d4AITn84kSOk06hjIv1y49/view?usp=sharing) and use *deprecated/whangpth2bin.py* to transform it into our form; compared to *bert-base-uncased*, only need to +1 for vocab size in *config.json* and add a new word [EOS] after *vocab.txt*
    * Or use our [pretrained models](https://www.dropbox.com/scl/fo/x6vtnwoj6luar7a6x4vl1/h?rlkey=o253jwdcz0qpu89idj76vi2tr&dl=0) (already transformed) instead
6. Reproduce BERT-VFT result
    ```bash
    ~/SL4DU/code/SL4DU$ python3 -u train.py --save_ckpt --task=RS --dataset=Ubuntu --pkl_data_path=../../data/ubuntu_data --pretrained_model=post-ubuntu-bert-base-uncased --freeze_layers=8 --train_batch_size=16 --eval_batch_size=100 --log_dir=? #--pkl_valid_file=test.pkl
    ```
6. Reproduce SL4RS result
    ```bash
    ~/SL4DU/code/SL4DU$ python3 -u train.py --save_ckpt --task=RS --dataset=Ubuntu --pkl_data_path=../../data/ubuntu_data --pretrained_model=post-ubuntu-bert-base-uncased --freeze_layers=8 --train_batch_size=4 --eval_batch_size=100 --log_dir=? --use_NSP --use_UR --use_ID --use_CD --train_view_every=80 #--pkl_valid_file=test.pkl
    ```
7. Evaluation
    ```bash
    ~/SL4DU/code/SL4DU$ python3 -u eval.py --task=Ubuntu --data_path=../../data/ubuntu_data --pretrained_model=post-ubuntu-bert-base-uncased --freeze_layers=8 --eval_batch_size=100 --log_dir ? --load_path=?
    ```

## Pretrained on yourself
Using [Whang's repo](https://github.com/taesunwhang/BERT-ResSel)<!-- or [our fork](https://github.com/RayXu14/BERT-ResSel).-->

Remember to transform the saved model to our form using *deprecated/whangpth2bin.py*.

### Additional information for pretraining settings
set the number of epochs as 2 for post-training with 10 duplication data and set the virtual batch size as 384
