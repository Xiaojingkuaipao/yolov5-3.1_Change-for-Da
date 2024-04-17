import numpy as np
import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
    Return:
      sum(kernel_val): 多个核矩阵之和
    '''
    # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
    n_samples = int(source.shape[0]) + int(target.shape[0])
    total = torch.cat([source, target], dim=0)
    # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
    total0 = total.unsqueeze(0).expand(
        int(total.shape[0]), int(total.shape[0]), int(total.shape[1]), int(total.shape[2]), int(total.shape[3])
    )
    # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按复制
    total1 = total.unsqueeze(1).expand(
        int(total.shape[0]), int(total.shape[0]), int(total.shape[1]),int(total.shape[2]), int(total.shape[3])
    )
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance_square.data) / (n_samples ** 2 - n_samples)
    # 多核MMD
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # print(bandwidth_list)
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # 多核合并


def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
     source: (b1,n)的X分布样本数组
     target:（b2，n)的Y分布样本数组
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
     loss: MK-MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 将核矩阵分成4部分
    loss = 0
    XX = torch.mean(kernels[:batch_size, :batch_size])
    YY = torch.mean(kernels[batch_size:, batch_size:])
    XY = torch.mean(kernels[:batch_size, batch_size:])
    YX = torch.mean(kernels[batch_size:, :batch_size])
    # 这里计算出的n_loss是每个维度上的MK-MMD距离，一般还会做均值化处理
    n_loss = torch.mean(XX + YY - XY - YX)
    return n_loss