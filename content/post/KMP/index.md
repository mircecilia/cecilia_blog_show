---
date : '2025-03-06T21:30:07+08:00'
draft : flase
title : 'KMP'
categories:
  - string
  - DSAA
  - C
---

## next数组的计算

``` c
void computenext (const char* pattern, int m, int *next){
    int len = 0;
    next [0] = 0;
    int i = 1;
    while(i<m){
        if(pattern[i]==pattern[len]){
            len++;
            next[i]=len;
            i++;
        }
        else{
            if(len!=0){
                len = next[len-1];
            }
            else{
                next[i]=0;
                i++;
            }
        }
    }
}
```

next 数组的计算本质是查询给定文本的前后缀是否相同，next 数组默认首位为零，以 i 为下标进行检测，i 指向的字母即本次检测字符，若相同则 len 与 i 加一，若不相同，则检测此时不加检测字母的上一次检测是否有相同前后缀（即此时 len 是否为零），若为零则将 i 后移一位继续进行检测，若不为零则回退检测进度  
此处对进度回退进行详细解释：若已确定 i 对应的检测字母不相同，那么此时若存在相同前后缀（即 next[i] != 0），则一定是 i 所指字母与前一位（或多位）字母构成了相同前后缀，无需使用暴力算法进行计算，只需查询去掉 i 所指字母的上一串具有相同前后缀的字符串的 next 数组就可以确定下一次要检测的更小的相同前后缀字符串，类似于对自身进行了递归调用，但采用的是哨兵记法，个人感觉更类似于动态规划切钢管时使用的哨兵版本自底向顶算法  
e.g:  
若给定文本是 "abdcabp abdcabe"，此时 i 指向 e，检测得到 p 不等于 e，那么只有可能是 "...ab(e)" 与前面形成了更短的相同前缀，此时由于在第一次检测 "abdcabd" 中已经检测过 "ab" 这个字符串的 next 数组，所以只需查找 "abdcab" 中第二次 "ab" 的 next 数组即可得到下一次需要检测的字符串，在本例中就是检测加入 i 所指字母后是否构成 "abd"

## KMP算法

``` c
void kmp (const char* text, const char* pattern){
    int i = 0;
    int j = 0;
    int n = strlen(pattern);
    int m = strlen(text);
    int next[n];
    computenext(pattern,n,next);
    while(i < m){
        if(text[i]==pattern[j]){
            i++;
            j++;
        }
        else if(j == n){
            printf("have found! the location is %d",i-j);
            j = next[j-1];
        }
        else if(i < n && text[i]!=pattern[j]){
            if(j != 0){
                j = next[j-1];
            }
            else{
                i++;
            }
        }
    }
}
```

KMP 算法使用 i 与 j 两个指针分别指向 text 与 pattern 文本，若检测 i 与 j 所指字母相同，则两个指针全部后移，如果 j 指针移动距离与 pattern 字符串长度相等，则返回找到的第一个匹配的位置，回退后继续寻找下一个位置，若两指针所指字母不同，则检测 j 指针是否移动过，若未移动过则证明已检测的 text 中完全不可能含有任何一个字母匹配，后移指针 i 继续检测，若 j 已移动过，则利用 next 数组将 j 指针回退到重叠部分继续检测  
e.g：  
若 text 为 "abaacababcac"，而 pattern 为 "ababc"，检测至第四位时两个字母不匹配，此时 j 指针按照 next 数组回退至 pattern 中第二位 "b"，跳过对第一位 "a" 的检测，即检测 "...(aa)..." 与 "ab" 是否相同，此时判断也不相同而无回退位置，从而检测出 text 中前四位 "abaa" 中完全无可能与 pattern 匹配，此时 i 指针移动至第五位开始新一轮检测

## 总结

KMP算法大幅度优化了BF算法的按位暴力回溯求解，通过保持指针 i 不动，使得时间复杂度由 O(m*n) 变为 O(m+n)，规避了坏情况对算法稳定度带来的影响
