# SL4DU

## Environment
### Option 1: container
frontlibrary/transformers-pytorch-gpu:4.6.1-pyarrow

```bash
~$ docker run --runtime=nvidia -it --rm -v $HOME/SL4DU:/workspace frontlibrary/transformers-pytorch-gpu:4.6.1-pyarrow
```

### Option 2: build from scatch
* Python==3.9 ([There may be some problem for numpy with 3.10](https://exerror.com/building-wheel-for-numpy-pyproject-toml/))
* numpy (can be automatically installed when installing scipy)
* scipy (if you have problem, see this [solution](https://stackoverflow.com/questions/11114225/installing-scipy-and-numpy-using-pip))
* torch==1.8
* pyarrow
* tqdm
* transformers==4.5.1
* sklearn


<!--#### 放弃的事情
1. jupyterlab
    1. 在大型项目中体验降低
    2. 各种网络限制
-->

## Reproduce step
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

<!-- 和论文不同的部分
* CD有问题但不知道问题是什么：放弃挣扎，改为分类式
    * 观察代码
        * 数据准备阶段
            * 不可能是被其他的任务干扰，因为每个任务都是独立从数据集deepcopy出来的样例
    * 实验验证
        * 在loss的scale修正后导致的，但经过确认，就算是恢复原本的loss计算方法仍然无效
            * 但用乘数来补偿scale无效，虽然理论上和修正之前等价
            * 增大补句范围无效
            * 拉高学习率会当场过拟合，而且据我看源代码Adam也并非线性关系
        * 扩大范围到同一个session的拿来用效果更差了一点点
        * 切换回Classification的经典样式后有效！
            * 限定为同speaker后略有下降但是仍然有效
            * 在此基础上改成用自己的方法土写的margin ranking loss，终于有效！莫非pytorch提供的接口我理解还是有误？
            * 试图写得更精简以及修正和论文不同的部分（去掉EOS），但是导致效果下降！
                * 很奇怪的是，增加句例导致修正后效果提升，但修正前变成无效。
                * 此任务很不稳定，切莫再行修改。维持现状即可。
                * 还是发现CD任务有很强的不稳定性-->

## Pretrained on yourself
Using [Whang's repo](https://github.com/taesunwhang/BERT-ResSel) or [our fork](https://github.com/RayXu14/BERT-ResSel).

Remember to transform the saved model to our form using *deprecated/whangpth2bin.py*.

We provide our [pretrained models](https://www.dropbox.com/sh/l9ityw69ls3qyyj/AAARoLxHAP4f4lJ-twJ8IDpia?dl=0) (already transformed)

### Additional information for pretraining settings
set the number of epochs as 2 for post-training with 10 duplication data and set the virtual batch size as 384


## P.S.
1. warmup和lr_decay未应用，不过目前不注重调参
2. 多GPU提高eval速度明显，但提高train速度不明显，可能是因为梯度优化过程无法并行，却占据大部分的时间
3. ID任务需要不定长的数组，不方便直接使用使用Tensor，因此也无法多进程加载数据，考虑到上一条，此问题可以不改进
