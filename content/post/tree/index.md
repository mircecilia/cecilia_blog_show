---
date: 2025-02-25T14:05:50+08:00
description: 
draft: flase
title: 二叉树初步
categories:
  - tree
  - DSAA
  - python
---

## 树的定义

``` python
def tree(label,branches=[]):  
    for branch in branches:
        assert is_tree(branch),'branches must be branch`
        return [label]+list(branches)
def label(tree):
    return tree[0]
def branches(tree):
    return tree[1:]
def is tree(tree):
    if type(tree)!=list or len(tree)<1:
        return False
        for branch in branches(tree):
            if not is_tree(branch):
                return False
        return True
def is_leaf(tree):
    return not branches(tree)
#test=tree(3,[tree(7),tree(9,[tree(6),tree(0)])])
#print(test)
```

此处对树，枝干以及树叶进行定义并检测

## 斐波那契树

``` python
def fib_tree(n):
    if n<=1:
        return tree(1)
    else:
        left,right=fib_tree(n-1),fib_tree(n-2)
        return tree(label(left)+label(right),[left,right]) 
#test=fib_tree(5)
#print(test)
```

此处可以通过递归自顶向下生成一颗斐波那契树

## 树叶计数

```python
def count_leaf(tree):
    if is_leaf(tree):
        return 1
    else:
        branches_count=[count_leaf(branch) for branch in branches(tree)] 
        #此处将不同枝干的数据作为列表进行储存，方便使用sum函数进行计算
        return sum(branches_count)
#test=count_leaf(fib_tree(5))
#print(test)
```

通过递归对树叶数量进行计算

## 树叶展示

``` python
def leaves(tree):
    if is_leaf(tree):
        return [label(tree)]
    else:
        return sum([leaves(b) for b in branches(tree)],[])
#test=leaves(fib_tree(5))
#print(test)
```

此处通过递归进行树叶显示，同时使用指定相加方式的 sum 函数将所有树叶以列表形式展示

## 树的修改

- 增加树叶

``` python
  def increment_leaves(t):
    if is_leaf(t):
        return tree(label(t)+1)
    else:
        bs = [increment_leaves(b) for b in branches(t)]
        return tree(label(t),bs)
```

此处通过递归检测输入是否为树叶，若是树叶则增加，若非树叶则向下继续检测

- 增加枝干

``` python
def increment(t):
    bs = [increment(b) for b in branches(t)]
    return tree(label(t)+1,bs)
```

检测原理同上，自顶向下

## 以可视化方式打印树

``` python
def print_tree(t,cnt=0):
    print(' '*cnt+str(label(t)))
    for b in branches(tree):
        print_tree(b,cnt+1)
```

此处通过控制空格数量将树的不同级枝干分开打印

## 检测树不同路径数之和

``` python
#t=tree(3,[tree(-1),tree(1,[tree(2,[tree(1)]),tree(3)]),tree(1,[tree(-1)])])
def count_paths(t,total):
    if label(t)==total:
        found = 1
    else:
        found = 0
    return found + sum([count_paths(b,total-label(t)) for b in branches(t)])
#test=count_paths(t,4)
#print(test)
```

此处仍运用递归，在经过每条路径时检测是否符合要求，若不符合则减去 label 值继续向下检测
