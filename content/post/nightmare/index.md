---
date : '2025-03-16T00:38:24+08:00'
draft : flase
title : '梦魇'
categories:
  - CSP
  - DSAA
  - C
---

## 代码内容

``` c
# include <stdio.h>

int findmin(int a,int b){
    return a > b ? b : a;
}

int findmax(int* arr,int n){
    int p = arr[0];
    for(int i = 0;i < n;i++){
        if(arr[i] >= p){
            p = arr[i];
        }
    }
    return p;
}

int compute(int* attack,int* defense,int n,int i,int max){
    int j = i;
    int k = i + 1;
    int need = 0;
    int attacknow = 0;
    if(defense[j] > defense[k]){
        need = defense[k];
        attacknow = need + attack[k];
        k++;
    }
    else{
        need = defense[j];
        attacknow = need + attack[j];
        j--;
    }
    while(j != -1 && k != n && defense[j] != max && defense[k] != max){
        if(attacknow >= findmin(defense[j],defense[k])){
            if(defense[j] > defense[k]){
                attacknow += attack[k];
                k++;
            }
            if(defense[j] <= defense[k]){
                attacknow += attack[j];
                j--;
            } 
        }
        else{
            if(defense[j] > defense[k]){
                need = need + defense[k] - attacknow;
                attacknow = need + attack[k];
                k++;
            }
            if(defense[j]<=defense[k]){
                need = need + defense[j] - attacknow;
                attacknow = need + attack[j];
                j--;
            }
        }
    }
    while((j == -1 || defense[j] == max) && (k != n && defense[k] != max)){
        if(attacknow < defense[k]){
            need = need + defense[k] - attacknow;
            attacknow = defense[k] + attack[k];
            k++;
        }
        else{
            attacknow += attack[k];
            k++;
        }
    }
    while((k == n || defense[k] == max) && (j != -1 && defense[j] != max)){
        if(attacknow < defense[j]){
            need = need + defense[j] - attacknow;
            attacknow = defense[j] + attack[j];
            j--;
        }
        else{
            attacknow += attack[j];
            j--;
        }
    }
    if((j == -1 || defense[j] == max) && (k == n || defense[k] == max)){
        if(attacknow < max){
            need = need + max - attacknow;
        }
        else{
            return need;
        }
    }
}

int output(int* attack,int* defense,int n){
    int p = 0;
    int max = findmax(defense,n);
    for(int i = 0;i < n-1;i++){
        int need = compute(attack,defense,n,i,max);
        p = p ^ need;
    }
    return p;
}

int main(){
    int n;
    scanf("%d",&n);
    int attack[n];
    int defense[n];
    for(int i = 0;i < n ;i++){
        scanf("%d",&defense[i]);
    }
    for(int i = 0;i < n;i++){
        scanf("%d",&attack[i]);
    }
    int t;
    scanf("%d",&t);
    int show[t];
    int cnt = 0;
    while(t > 0){
        int q;
        scanf("%d",&q);
        if(q == 0){
            show[cnt] = output(attack,defense,n);
            cnt++;
        }
        else{
            int stored[n];
            int storea[n];
            for(int i = 0;i < n;i++){
                stored[i] = defense[i];
                storea[i] = attack[i];
            }
            while(q > 0){
                int number, a, d;
                scanf("%d", &number);
                scanf("%d", &d);
                scanf("%d", &a);
                storea[number - 1] = a;
                stored[number - 1] = d;
                q--;
            }
            show[cnt] = output(storea,stored,n);
            cnt++;
        }
        t--;
    }
    for(int i = 0;i < cnt - 1;i++){
        printf("%d\n",show[i]);
    }
    printf("%d",show[cnt-1]);
    return 0;
}
```

## 思路概述

本题的主要思路是双指针，在插入位置的左侧及右侧分别置入 j , k 两个指针，然后检测此时所需攻击力是否大于左右指针指向的任意一个数，若有则击败其中最小的一个，并将所得攻击力加入 attacknow 中，若无，则补上 need 与所需最小攻击力（即最小防御力）的差值，并将 attacknow 设置为该指针指向的梦魇防御力与所得攻击力的和，重复检测直至任意一个指针指向最大值或到达数组边界，再让剩余的一个指针同样到达等效边界（即指向最大值或数组边界），由于能击败梦魇中最高防御力的敌人则证明能击败所有梦魇，所以只要两个指针都指向等效边界，即可检测此时 attacknow 是否能击败防御力最高的敌人，若能则直接返回 need 值，若不能则补全 attacknow 与最高防御力间的差值，然后返回 need 值即可，剩余的都是套壳，注意 coderunner 不能正常返回套壳后的结果，需要先存放至数组中再进行打印
