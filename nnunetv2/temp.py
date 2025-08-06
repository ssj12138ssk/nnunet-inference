import os
import shutil

def rename_files(directory):
    for filename in os.listdir(directory):
        if "_pred" in filename:
            new_name = filename.replace("_pred", "")
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

# rename_files("/data/data1/zxy/ChangX/AIDataset/HipDataset/mask-temp/")  # 指定你的目录



# 定义两个文件夹的路径
# dir_A = "/data/data1/zxy/ChangX/AIDataset/HipDataset/image"
# dir_B = "/data/data1/zxy/ChangX/AIDataset/HipDataset/mask"

# # 获取两个文件夹中的文件列表
# files_in_A = set(os.listdir(dir_A))
# files_in_B = set(os.listdir(dir_B))

# # 找出在 A 中但不在 B 中的文件
# files_to_delete = files_in_A - files_in_B

# # 删除这些文件
# for file in files_to_delete:
#     file_path = os.path.join(dir_A, file)
#     if os.path.isfile(file_path):
#         os.remove(file_path)
#     elif os.path.isdir(file_path):
#         shutil.rmtree(file_path)


# import os
# import shutil

# # 定义文件夹路径
# folder_A = '/data/data1/zxy/data/nnUnet_raw/Dataset701_Hip/image/'
# folder_B = '/data/data1/zxy/data/nnUnet_raw/Dataset701_Hip/labelsTs/'
# folder_C = '/data/data1/zxy/data/nnUnet_raw/Dataset701_Hip/imagesTs/'

# # 获取文件夹A和B中的文件名
# files_in_A = set(os.listdir(folder_A))
# files_in_B = os.listdir(folder_B)

# # 遍历文件夹B中的文件
# for file in files_in_B:
#     # 如果文件也在文件夹A中，则复制到文件夹C
#     if file in files_in_A:
#         print(file)
#         shutil.copy(os.path.join(folder_A, file), folder_C)


# import os
# import nibabel as nib
# import numpy as np

# # 指定文件夹路径
# folder_path = '/data/data1/zxy/data/nnUnet_raw/Dataset701_Hip/labelsTr/'

# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     # 只处理 .nii.gz 文件
#     if filename.endswith('.nii.gz') and filename.startswith("Clinical_case_"):
#         file_path = os.path.join(folder_path, filename)
        
#         # 加载图像
#         img = nib.load(file_path)
#         data = img.get_fdata()
        
#         # 检查图像的最大值
#         if data.max() == 5:
#             # 创建一个副本，以避免改变原始数据
#             new_data = data.copy()
            
#             # 将1变为3，将3、4、5变为1
#             new_data[data == 1] = 3
#             new_data[(data >= 3) & (data <= 5)] = 1
            
#             # 创建新的NIfTI图像
#             new_img = nib.Nifti1Image(new_data, img.affine, img.header)
            
#             # 保存新的NIfTI图像
#             new_file_path = os.path.join(folder_path, filename)
#             print(new_file_path)
#             nib.save(new_img, new_file_path)


# import os
# import glob

# def rename_files(directory):
#     # 使用glob模块找到所有包含"_0000"的文件
#     for filename in glob.glob(os.path.join(directory, "*_0000*")):
#         # 使用os模块获取文件的路径和新的文件名（即去掉"_0000"）
#         new_filename = os.path.join(directory, os.path.basename(filename).replace("_0000", ""))
#         # 重命名文件
#         os.rename(filename, new_filename)

# # 指定你的目录
# directory = "/data/data1/zxy/data/nnUnet_raw/Dataset701_Hip/labelsTs/"
# rename_files(directory)

# import os
# import glob
# import nibabel as nib
# import numpy as np
# import gzip

# def round_nii_files(directory):
#     # 使用glob模块找到所有以"Clinical_case"开头的nii.gz文件
#     for filename in glob.glob(os.path.join(directory, "Clinical_case*.nii.gz")):
#         try:
#             # 使用nibabel库读取nii.gz文件
#             print(filename)
#             img = nib.load(filename)
#             data = img.get_fdata()
            
#             # 将图像中的数值四舍五入到最近的整数并转换为uint8类型
#             rounded_data = np.round(data).astype(np.uint8)
            
#             # 创建新的nii图像
#             new_img = nib.Nifti1Image(rounded_data, img.affine, img.header)
            
            
#             # 保存修改后的nii.gz文件
#             nib.save(new_img, filename)
        
#         except gzip.BadGzipFile:
#             print(f"File {filename} is corrupted and was skipped.")

# # 指定你的目录
# directory = "/data/data1/zxy/data/nnUnet_raw/Dataset701_Hip/labelsTs/"
# round_nii_files(directory)

import numpy as np
import nibabel as nib
import SimpleITK as sitk

# Load npy file
data = np.load("/data/data1/zxy/data/nnUnet_preprocessed/Dataset701_Hip/nnUNetPlans_3d_fullres/HB-ZhongshanHosp_case_0014_Hip_seg.npy")

# Create a Nifti1Image
img = sitk.GetImageFromArray(data)

# Save as .nii.gz
sitk.WriteImage(img, '/data/data1/zxy/data/nnUnet_preprocessed/Dataset701_Hip/nnUNetPlans_3d_fullres/HB-ZhongshanHosp_case_0014_Hip_seg.nii,gz')