import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from transvae.tvae_util import *

def vae_data_gen(mols, props, char_dict):
    """
    将输入的SMILES字符串编码为带有标记ID的张量。
    参数：
        mols (np.array, 必需): 包含分子结构的数组
        props (np.array, 必需): 包含标量化学属性值的数组
        char_dict (字典, 必需): 将标记映射到整数ID的字典
    返回：
        encoded_data (torch.tensor): 包含每个SMILES字符串编码的张量
    """
    #print(mols)
    smiles = mols[:,0]  # 提取SMILES字符串
    #print(smiles)
    if props is None:  # 如果未提供属性，创建一个全零的属性数组
        props = np.zeros(smiles.shape)
    del mols  # 删除原始的分子数组以节省内存
    smiles = [tokenizer(x) for x in smiles]  # 对SMILES字符串进行分词
    #print(smiles)
    print(len(smiles))
    encoded_data = torch.empty((len(smiles), 136))  # 创建一个空的编码数据张量,其中224为SMILES分子式中token数量最长的
    #print(encoded_data)
    for j, smi in enumerate(smiles):
        #encoded_smi = encode_smiles(smi, 126, char_dict)  # 编码SMILES字符串
        encoded_smi = encode_smiles(smi, 134, char_dict)  # 编码SMILES字符串
        encoded_smi = [0] + encoded_smi  # 在编码前添加起始标记
        #print(len(encoded_smi))
        encoded_data[j,:-1] = torch.tensor(encoded_smi)  # 填充编码数据张量
        
        #print(props[j])
       # encoded_data[j,-1] = torch.tensor(props[j])  # 添加属性值
       # print("encoded_data shape:", encoded_data.shape)
        #print("props[j] shape:", props[j].shape)
        #encoded_data[j,-1] = torch.tensor(props[j])
        
        encoded_data[j,-1] = torch.tensor(props[j][0], dtype=torch.float)  # 将第一个属性值转换为张量，并添加到encoded_data张量的最后一列中
        encoded_data[j,-2] = torch.tensor(props[j][1], dtype=torch.float)  # 将第二个属性值转换为张量，并添加到encoded_data张量的倒数第二列中
        encoded_data[j,-3] = torch.tensor(props[j][2], dtype=torch.float)
        encoded_data[j,-4] = torch.tensor(props[j][3], dtype=torch.float)
        encoded_data[j,-5] = torch.tensor(props[j][4], dtype=torch.float)
        encoded_data[j,-6] = torch.tensor(props[j][5], dtype=torch.float)
        encoded_data[j,-7] = torch.tensor(props[j][6], dtype=torch.float)
        
        if props[j][7]=='nan':
            props[j][7]=0
        encoded_data[j,-8] = torch.tensor(props[j][7], dtype=torch.float)
        if props[j][8]=='nan':
            props[j][8]=0
            #encoded_data[j,-8] = torch.tensor(props[j][7], dtype=torch.float)
        encoded_data[j,-9] = torch.tensor(props[j][8], dtype=torch.float)
        #print(encoded_data)
    return encoded_data  # 返回编码后的数据

def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)

    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
