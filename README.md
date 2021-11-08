# SL4DU

## 环境
### 选项一：容器环境
frontlibrary/transformers-pytorch-gpu:4.6.1-pyarrow

```bash
~$ docker run --runtime=nvidia -it --rm -v $HOME/SL4DU:/workspace frontlibrary/transformers-pytorch-gpu:4.6.1-pyarrow
```

### 选项二：自建环境
* Python==3.9 ([3.10安装一些包目前可能出问题](https://exerror.com/building-wheel-for-numpy-pyproject-toml/))
* numpy (自动由scipy依赖安装)
* scipy (可能遇到问题->[解决方案](https://stackoverflow.com/questions/11114225/installing-scipy-and-numpy-using-pip))
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

## 复现步骤
1. 初始化文件夹结构
    ```
    SL4DU
        code
        data
        pretrained
    ```
2. 下载代码和Ubuntu数据
    *
    ``` bash
    ~/SL4DU/code$ git clone https://github.com/RayXu14/SL4DU.git
    ~/SL4DU/data$ wget https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip
    ~/SL4DU/data$ unzip ubuntu_data.zip
    ```
3. 往pretrained文件夹放[bert-base-uncased](https://huggingface.co/bert-base-uncased/tree/main)预训练模型进行测试
    * config.json
    * vocab.txt
    * pytorch_model.bin
3. 预处理数据
    ```bash
    ~/SL4DU/code/SL4DU$ python3 preprocess.py --task=RS --dataset=Ubuntu --raw_data_path=../../data/ubuntu_data --pkl_data_path=../../data/ubuntu_data --pretrained_model=bert-base-uncased
    ```
4. BERT复现
    ```bash
    ~/SL4DU/code/SL4DU$ python3 -u train.py --save_ckpt --task=RS --dataset=Ubuntu --pkl_data_path=../../data/ubuntu_data --pretrained_model=bert-base-uncased --add_EOT --freeze_layers=0 --train_batch_size=8 --eval_batch_size=100 --log_dir=? # --pkl_valid_file=test.pkl
    ```
5. 往pretrained文件夹添加post-ubuntu-bert-base-uncased
    * 放置[whang的Ubuntu的ckpt](https://drive.google.com/file/d/1jt0RhVT9y2d4AITn84kSOk06hjIv1y49/view?usp=sharing)，用deprecated/whangpth2bin.py转化为可用库自动加载的格式；和bert-base-uncased相比，config.json的改动仅需要词汇比post adaption之前的+1，词汇表在末尾新开一行增加[EOS]
6. BERT-VFT复现
    ```bash
    ~/SL4DU/code/SL4DU$ python3 -u train.py --save_ckpt --task=RS --dataset=Ubuntu --pkl_data_path=../../data/ubuntu_data --pretrained_model=post-ubuntu-bert-base-uncased --freeze_layers=8 --train_batch_size=16 --eval_batch_size=100 --log_dir=? #--pkl_valid_file=test.pkl
    ```
6. SL4RS复现
    ```bash
    ~/SL4DU/code/SL4DU$ python3 -u train.py --save_ckpt --task=RS --dataset=Ubuntu --pkl_data_path=../../data/ubuntu_data --pretrained_model=post-ubuntu-bert-base-uncased --freeze_layers=8 --train_batch_size=4 --eval_batch_size=100 --log_dir=? --use_NSP --use_UR --use_ID --use_CD --train_view_every=80 #--pkl_valid_file=test.pkl
    ```
7. 测试
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

## P.S.
1. warmup和lr_decay未应用，不过目前不注重调参
2. 多GPU提高eval速度明显，但提高train速度不明显，可能是因为梯度优化过程无法并行，却占据大部分的时间
3. ID任务需要不定长的数组，不方便直接使用使用Tensor，因此也无法多进程加载数据，考虑到上一条，此问题可以不改进
