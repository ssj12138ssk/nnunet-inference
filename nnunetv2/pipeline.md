# nnUnet pipeline
ibot版nnUnet相比原版，将resample从skimage变为sitk，与C++的itk保持一致
1. 制作nnUnetRaw, nnUnet_preprocessed, nnUnet_results三个文件夹
2. 在nnUnet文件夹里面制作数据集文件夹，如Dataset701_Hip，以Dataset开头，701是数据集的索引
3. 在数据集文件夹里面建立 imageTr, imageTs, labelsTr, labelsTs四个子文件夹，然后将对应的训练和测试图片放进去，image最后要加_0000来区分模态，label不要加_0000；制作dataset.json 
4. 在bashrc里面添加环境变量
```
export nnUNet_raw="/data/data1/zxy/data/nnUnet_raw"
export nnUNet_preprocessed="/data/data1/zxy/data/nnUnet_preprocessed"
export nnUNet_results="/data/data1/zxy/data/nnUnet_results"
```
5. 激活环境变量 source /home/user/.bashrc
6. 预处理 nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity，若遇到label与dataset.json 设置的label不一致的情况要及时处理
7. 训练 
CUDA_VISIBLE_DEVICES={X} nnUNetv2_train {task_id} 3d_fullres {4/all}
如果身体多部位进行分割，有一些部位是对称的，所以训练的时候增强不能mirror
CUDA_VISIBLE_DEVICES={X} nnUNetv2_train {task_id} 3d_fullres {4/all} -tr nnUNetTrainerNoMirroring
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 701 3d_fullres all -tr nnUNetTrainerNoMirroring
8. 如果训练开始报错mmap length is greater than file size ，则删掉pre_processing里面的npy文件
8. screen
```
新建会话 
screen -S 会话名字 
重新进入断联或关闭的会话 
screen -r 会话名
screen -d -r 会话名
删除
若不在该会话内 # screen -S 会话名 -X quit
若在该会话中可直接 # exit
```
10. 推理
```
nnUNetv2_predict -i {input_dir} -o {output_dir} -d {task_id} -c 3d_fullres --save_probabilities -f {4/all} --disable_tta
--disable_tta：不use_mirroring

CUDA_VISIBLE_DEVICES=3 nnUNetv2_predict -i /data/data1/zxy/data/nnUnet_raw/Dataset701_Hip/imagesTs/ -o /data/data1/zxy/data/nnUnet_results/Dataset701_Hip/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_all/test/ -d 701 -c 3d_fullres --save_probabilities -f all --disable_tta -tr nnUNetTrainerNoMirroring
```
nnUNetv2_predict -i /data/data1/zxy/data/temp/ -o /data/data1/zxy/data/tempO/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_all/test/ -d 701 -c 3d_fullres -f all --disable_tta -tr nnUNetTrainerNoMirroring