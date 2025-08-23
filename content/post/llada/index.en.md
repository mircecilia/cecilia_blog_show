---
date : '2025-08-22T12:19:40+08:00'
draft : flase
title : 'LLaDA'
categories:
  - Pytorch
  - dllm
---

## 基于高斯分布的噪声添加

``` python
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
```
此函数强制转化 logits 数据类型为 float64 ，并将基于高斯分布的噪声转化为与 logits 相同的形状，最终将 logits 转化为概率形式，通过除以 gumbei_noise 加入噪声

## 计算每步解码的 token 数量

``` python
def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens
```
此函数接收 mask_token 的布尔序列以及解码步数，使用 sum 函数获取掩码 token 数量，再计算每步的基础解码数与无法均摊的余数，最终返回每步需要解码的 token 数

## 初始化与块划分

```python
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length # gen_length 为生成 token 的长度

    assert steps % num_blocks == 0
    steps = steps // num_blocks
```
本段函数先将 x 的 token 全部初始化为 mask token（此处规定为126336），然后再将 prompt 的 token 部分进行填充，之后对块进行划分，再计算每个块需要迭代多少步

## 块内解码

```python
for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
```
本段函数先获取块内掩码序列，然后通过调用 get_num_transfer_tokens 函数计算块内解码 token 数，对于每一步，获取 mask_index ，然后判断是否需要条件预测，在 llada 源码中并不需要，但可以进行修改，若进行条件预测，则复制一份 un_x ，并将 prompt 部分替换为 mask_token（126336），并将 x 与 un_x 进行拼接，然后计算 logits ，再将其拆分有条件预测与无条件预测的 logits 与 un_logits ,最后使用 CFG 公式进行计算，对条件结果进行强化，若不启用 CFG ，则直接计算 logits ，之后调用 add_gumbel_noise 函数对刚刚计算出的 logits 添加噪声，再取 argmax 在 vocab_size 上算出候选 token 序列 x0

## 解码方式选择

```python
if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
```
若采用置信度解码方式，则通过 softmax 函数将 logits 转化为概率分布，将每个 token 的置信度写入 x0_p 中

若采用随机解码，则随机生成每个 token 的置信度写入 x0_p 中

## 屏蔽范围外及已解码的 token 

```python
x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

x0 = torch.where(mask_index, x0, x)
confidence = torch.where(mask_index, x0_p, -np.inf)
```
本段函数首先将超出该块位置范围的 token 置信度设为 -inf 避免被选，然后将掩码部分的 token 用置信度代替，并保持已解码 token 不变，将该序列写入 x0 ，之后将已解码部分的置信度设为 -inf 写入 confidence

## 更新原序列 x

```python
transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
for j in range(confidence.shape[0]):
    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
    transfer_index[j, select_index] = True
x[transfer_index] = x0[transfer_index]
```
本段函数先将 transfer_index 设置为 False ，然后在每批次样本中选择置信度最高的 token ，将其位置写入 select_index 中，然后将 transfer_index 中对应位置修改为 True ，用其将解码后的 token 写入原序列，覆盖 mask_id

## generate example

- original:
```
    [
    [-1,-1,-1,-1,-1]
    ,

    [-1,-1,-1,-1,-1]
    ]
```
- add prompt:
```  
    [
    [3795,-1,-1,-1,-1]
    ,

    [2294,-1,-1,-1,-1]
    ]
```
- calculate:
  
    step -> 4 -> 2

    gen_length -> 4

    block_length -> 2
```
    mask_index -> 
    
    [
    [False,True,True,True,True]
    ,

    [False,True,True,True,True]
    ]
```
- select one of the block : **(the first)**
```  
    block_mask_index -> 
    
    [
    [False,True,True,False,False]
    ,

    [False,True,True,False,False]
    ]
```

```
    num_transfer_tokens -> 
    
    [
    [1,1]
    ,

    [1,1]
    ]
```
- the first step : 
```
    mask_index -> 
    
    [
    [False,True,True,False,False]
    ,

    [False,True,True,False,False]
    ]

    logits.shape -> (2 , 5 , 5000 (size of vocabulary))
```
- add noise : 
```
    logits_with_noise.shape -> (2 , 5 , 5000 (size of vocabulary))

    x0 -> 
    [
    [1978,2458,2049,1473,2659]
    ,

    [1654,2521,3364,1758,3867]
    ]
```
- remasking : 
```  
    p -> 

    x0_p -> 
    [
    [0.87,0.95,0.14,0.36,0.47]
    ,

    [0.45,0.26,0.41,0.36,0.57]
    ] -> 基于 softmax 的置信度
```
- random : 
```
    x0_p -> 
    [
    [0.87,0.95,0.14,0.36,0.47]
    ,

    [0.45,0.26,0.41,0.36,0.57]
    ] -> 随机置信度
```
- update ； 
```  
    x0_p -> 
    [
    [-inf,0.95,0.14,-inf,-inf]
    ,

    [-inf,0.26,0.41,-inf,-inf]
    ]

    x0 -> 
    [
    [1978,0.95,0.14,1473,2659]
    ,

    [1654,0.26,0.41,1758,3867]
    ]

    confidence -> 
    [
    [-inf,0.95,0.14,-inf,-inf]
    ,

    [-inf,0.26,0.41,-inf,-inf]
    ]
```

- overwrite x
```
    transfer_index -> 
    [
    [False,False,False,False,False]
    ,

    [False,False,False,False,False]
    ]

    select_index -> tensor([1],[2])

    transfer_index -> 
    [
    [False,True,False,False,False]
    ,

    [False,False,True,False,False]
    ]

    x ->
    [
    [3795,2458,-1,-1,-1]
    ,

    [2294,-1,3364,-1,-1]
    ]
```
