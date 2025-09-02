---
date : '2025-08-29T11:18:33+08:00'
draft : flase
title : 'Dream'
categories:
  - Pytorch
  - dllm
---

## 基于概率阈值保留候选 token

###  top_p

```python
def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits
```
本函数先将得到的 logits 进行递减排序并获取索引，然后计算累加和，超过 top_p 概率的 logits 将被标记为 True，同时通过将整个列表右移并在第一位填充 0 来强制保留概率最大的候选 token，创建一个全为 Flase 的 mask 列表，将上述修改后的索引映射回去，最后将 mask 列表中标记为 True 的位置的 logits 设置为极小值，使其不会被采样

> **generate example**

```python
候选 token -> [A, B, C, D]
logits -> [2.0, 0.5, 1.0, -0.5]
top_p -> 0.8
sorted_logits -> [2.0, 1.0, 0.5, -0.5]
sorted_indices -> [0, 1, 2, 3]
softmax :
probs -> [0.57, 0.21, 0.14, 0.08]
累计和 -> [0.57, 0.78, 0.92, 1.00]
sorted_indices_to_remove -> [False, False, True, True]
-> [False, False, False, True]
mask -> [False, False, False, True]
logits -> [2.0, 1.0, 0.5, -1e38]
```

### top_k

```python
def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits
```
本函数先进行安全性检查，防止 top_k 的大小大于词表大小，然后将前 k 大的 logits 设置为极小值防止其被采样

> **generate example**

```python
logits -> [2.0, 1.0, 0.5, -0.5]
top_k -> 2
indices_to_remove -> [False, False, True, True]
logits -> [2.0, 1.0, -1e38, -1e38]
```

## 从候选 token 中采样

```python
def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0
```
该函数根据温度大小控制 logits 分布的尖锐程度，然后应用 top_k / top_p 策略过滤可能性低的 token，之后使用 softmax 将 logits 归一化为概率分布，若温度大于零，则使用多项分布采样，按照各 logits 概率选出解码 token，否则直接选取概率最大的 token

若使用 margin_confidence，则置信度定义为第一名概率减去第二名概率，若使用 neg_entropy，则置信度定义为负熵，即分布越尖锐置信度越高

> **generate example**

```python
采用 top_k / top_p 过滤后 :
logits -> [2.0, 1.0, 0.5]

temperture -> 0 :
softmax -> [0.62, 0.23, 0.15]
选择 token -> A
confidence -> 0.62 ; x0 -> 0

temperture -> 2 :
logits -> [1.0, 0.5, 0.25]
softmax -> [0.48, 0.29,0.23] # 分布更平缓
选择 token -> A / B / C (按照概率随机抽样)

confidence :
margin_confidence -> 0.19 ; x0 -> 0 / 1 / 2
neg_entropy -> -1.05 ; x0 -> 1
```

## 离散扩散解码

```python
def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None
```
该段负责初始化，从 generation_config 中读取相应的参数，并创建一个空列表保存中间的采样过程

```python
x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

if attention_mask is not None and torch.any(attention_mask == 0.0):
    attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
    tok_idx = attention_mask.long().cumsum(-1) - 1
    tok_idx.masked_fill_(attention_mask == 0, 1)
    attention_mask = torch.logical_and(
        attention_mask.unsqueeze(1).unsqueeze(-2),
        attention_mask.unsqueeze(1).unsqueeze(-1),
    )
else:
    tok_idx = None
    attention_mask = "full"
```
该段将输入序列的右侧全部使用 mask_token_id 填充，同时判断是否进行注意力遮蔽，若进行注意力遮蔽，则扩展传入的 attention_mask 至 max_length ，然后生成符合 transformer 架构的四维张量，使注意力机制只在非遮蔽 token 上进行，同时生成有效 token 位置相应的索引

> **generate example**

```python
attention_mask -> [1, 1, 0, 1, 0]
max_length -> 8
attention_mask -> [1, 1, 0, 1, 0, 1, 1, 1]
tok_idx -> [0, 1, 1, 2, 2, 3, 4, 5]
-> [0, 1, 1, 2, 1, 3, 4, 5]

attention_mask ->
[1, 1, 0, 1, 0, 1, 1, 1],
[1, 1, 0, 1, 0, 1, 1, 1],
[0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 1, 0, 1, 1, 1],
[0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 1, 0, 1, 1, 1],
[1, 1, 0, 1, 0, 1, 1, 1],
[1, 1, 0, 1, 0, 1, 1, 1]
```

```python
timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
x = generation_tokens_hook_func(None, x, None)
for i in range(steps):
    mask_index = (x == mask_token_id)
    logits = self(x, attention_mask, tok_idx).logits
    logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
    logits = generation_logits_hook_func(i, x, logits)

    mask_logits = logits[mask_index]
    t = timesteps[i]
    s = timesteps[i + 1]
      
    if alg == 'origin':
        p_transfer = 1 - s / t if i < steps - 1 else 1
        x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
        transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
        _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
        x[mask_index] = x0.clone()
    else:
        if alg == 'maskgit_plus':
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
        elif alg == 'topk_margin':
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
        elif alg == 'entropy':
            confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
        else:
            raise RuntimeError(f"Unknown alg: {alg}")
```
本段函数先生成从 1 递减至 eps 的时间序列，然后调用前向传播函数生成 logits 并整体进行右移，以便对齐预测 token 位置，计算交叉熵 loss

> **origin** : 先计算替换比例 p_transfer，之后在每个位置上随机获得概率，低于该概率的位置被填充为 True，之后调用 sample_tokens，在每个 True 位置上，根据所选算法生成被采样的 token，最后对 x 进行更新

```python
num_mask_token = mask_index.sum() / mask_index.shape[0]
number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
full_confidence[mask_index] = confidence
if number_transfer_tokens > 0:
    if alg_temp is None or alg_temp == 0:
        _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
    else:
        full_confidence = full_confidence / alg_temp
        full_confidence = F.softmax(full_confidence, dim=-1)
        transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
    x_[mask_index] = x0.clone()
    row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
    x[row_indices,transfer_index] = x_[row_indices,transfer_index]
```
该段函数计算本步解码的 token 数，同时将置信度填在解码位置上，并将其余位置的置信度设置为 -inf，若使用 softmax 解码，则将置信度进行归一化，使用其概率进行随机抽样解码，否则选择置信度最高的位置进行解码，最后使用采样结果 x0 覆写 x 

```python
    x = generation_tokens_hook_func(i, x, logits)
    if histories is not None:
        histories.append(x.clone())
        
    if return_dict_in_generate:
        return DreamModelOutput(
            sequences=x,
            history=histories,
            )
    else:
        return x
```
该段函数通过 histories 保存并查看解码过程，返回最终的生成序列

## 归一化 embedding 后的向量

```python
class DreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```
本段函数先保存输入 dtype，然后将其转化为 float32 以保证数值稳定，随后对 embedding 后的向量求均方根，保证向量不改变方向而只改变长度，不损害基本语义

> **generate example**
```python
x -> [1.0, 2.0, 3.0, 4.0]
forward : 
x -> [0.1826, 0.3651, 0.5477, 0.7303]
```

## ROPE 频率表计算

```python
@torch.no_grad()
def forward(self, x, position_ids):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```
ROPE 所对应的函数部分先进行初始化，同时添加重置参数，动态更新两个辅助函数，在此不做摘录，此段函数是 ROPE 的频率表实现，先复制 self.inv_freq 至每份 batch，然后对 position_ids 进行扩张与转置，使其可以同 inv_freq_expanded 做矩阵乘法，二者相乘后转置，拼接，即可得到每个样本在对应的位置，维度需要旋转的角度

> - 在此说明为什么要对 emb 表进行复制，拼接 ：
> 
>   由于 ROPE 的原理是每两个维度分为一组，因此在先前计算 freq 表时，实际上是在计算每两个维度分组后对应的角度，故得到的频率表需要复制后拼接才能对应原先的 hidden 层维度
> - 对于一个样本的某个 token 向量，频率表计算方法如下：
>
>   inv_freq_expanded -> [[1.0], [0.1], [0.01]]
> 
>   position_ids_expanded -> [3]
> 
>   freqs -> [3.0, 0.3, 0.03]
> 
>   -> [3.0, 0.3, 0.03, 3.0, 0.3, 0.03]
> - 那么为什么不是 [3.0, 3.0, 0.3, 0.3, 0.03, 0.03] 呢？
> 
>   事实上这两种写法的效果是完全等效的，但是 PyTorch 在实现矩阵旋转的时候，是通过逐元素相乘和重排的方法来实现的，这样做会更方便进行广播，这点我们在后面再进行详细解析

## ROPE 

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
本段函数是 ROPE 的核心实现，rotate_half 先将 embedding 后的向量 (hidden state) 拆成前后两半，对后半取反后拼接，将整个向量 (x1, x2) 视作复数 (x1 + x2 i) 进行九十度旋转

apply_rotary_pos_emb 将 cos 与 sin 进行升维广播，之后调用  rotate_half将 Q、K 的各维度成对进行旋转，最终返回添加位置信息后的 Q、K 向量

> **generate example**

```python
rotate_half :
x -> [1.0, 2.0, 3.0, 4.0]
-> [-3.0, -4.0, 1.0, 2.0]
apply_rotary_pos_emb :
Q -> [1.0, 2.0, 3.0, 4.0]
cos -> [0.5, 0.5, 0.5, 0.5]
sin -> [0.866, 0.866, 0.866, 0.866]
rotate_half(Q) -> [-3.0, -4.0, 1.0, 2.0]
Q * cos -> [0.5, 1.0, 1.5, 2.0]
rotate_half(Q) * sin -> [-2.598,-3.464,0.866,1.732]
Q_embed -> [-2.098, -2.464, 2.366, 3.732]
```