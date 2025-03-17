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

本题的主要思路是双指针，在插入位置的左侧及右侧分别置入 j , k 两个指针，然后将 need 值设为两指针指向的两个梦魇中的最小防御力，移动指针之后重复检测此时所需攻击力是否大于左右指针指向的任意一个数，若有则击败其中最小的一个，并将所得攻击力加入 attacknow 中，若无，则补上 need 与所需最小攻击力（即最小防御力）的差值，并将 attacknow 设置为该指针指向的梦魇防御力与所得攻击力的和，之后重复上述过程，直至任意一个指针指向最大值或到达数组边界，再让剩余的一个指针同样到达等效边界（即指向最大值或数组边界），由于能击败梦魇中最高防御力的敌人则证明能击败所有梦魇，所以只要两个指针都指向等效边界，即可检测此时 attacknow 是否能击败防御力最高（必在其中一个等效边界）的敌人，若能则直接返回 need 值，若不能则补全 attacknow 与最高防御力间的差值，然后返回 need 值即可，剩余的都是套壳，注意 coderunner 不能正常返回套壳后的结果，需要先存放至数组中再进行打印

## 输入优化

``` c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUF_SIZE (1 << 20)
char buffer[BUF_SIZE];
size_t pos = 0, size = 0;

char get_char() {
    if (pos >= size) {
        size = fread(buffer, 1, BUF_SIZE, stdin);
        pos = 0;
        if (size == 0) return EOF;
    }
    return buffer[pos++];
}

int read_int() {
    int x = 0, c = get_char();
    while (c <= ' ' && c != EOF) c = get_char(); // 跳过空白字符直到有效输入或结束
    if (c == EOF) return -1; // 输入结束，返回特殊值
    do {
        x = x * 10 + (c - '0');
    } while ((c = get_char()) >= '0' && c <= '9');
    return x;
}

int findmin(int a, int b) {
    return a > b ? b : a;
}

int findmax(int* arr, int n) {
    int p = arr[0];
    for (int i = 0; i < n; i++) {
        if (arr[i] >= p) p = arr[i];
    }
    return p;
}

int compute(int* attack, int* defense, int n, int i, int max) {
    int j = i, k = i + 1;
    int need = 0, attacknow = 0;
    if (k >= n) return 0; // 防止越界
    if (defense[j] > defense[k]) {
        need = defense[k];
        attacknow = need + attack[k++];
    } else {
        need = defense[j];
        attacknow = need + attack[j--];
    }
    while (j >= 0 && k < n && defense[j] != max && defense[k] != max) {
        if (attacknow >= findmin(defense[j], defense[k])) {
            if (defense[j] > defense[k]) attacknow += attack[k++];
            else attacknow += attack[j--];
        } else {
            if (defense[j] > defense[k]) {
                need += defense[k] - attacknow;
                attacknow = need + attack[k++];
            } else {
                need += defense[j] - attacknow;
                attacknow = need + attack[j--];
            }
        }
    }
    while (k < n && defense[k] != max) {
        if (attacknow < defense[k]) {
            need += defense[k] - attacknow;
            attacknow = defense[k];
        }
        attacknow += attack[k++];
    }
    while (j >= 0 && defense[j] != max) {
        if (attacknow < defense[j]) {
            need += defense[j] - attacknow;
            attacknow = defense[j];
        }
        attacknow += attack[j--];
    }
    return (attacknow < max) ? need + max - attacknow : need;
}

int output(int* attack, int* defense, int n) {
    int p = 0;
    int max = findmax(defense, n);
    for (int i = 0; i < n - 1; i++) {
        p ^= compute(attack, defense, n, i, max);
    }
    return p;
}

int main() {
    int n = read_int();
    int* attack = (int*)malloc(n * sizeof(int));
    int* defense = (int*)malloc(n * sizeof(int));
    int* stored = (int*)malloc(n * sizeof(int));
    int* storea = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        defense[i] = read_int();
    }
    for (int i = 0; i < n; i++) {
        attack[i] = read_int();
    }
    int t = read_int();
    int* show = (int*)malloc(t * sizeof(int));
    int cnt = 0;
    while (t--) {
        int q = read_int();
        if (q == 0) {
            show[cnt++] = output(attack, defense, n);
        } else {
            memcpy(stored, defense, n * sizeof(int));
            memcpy(storea, attack, n * sizeof(int));
            while (q--) {
                int number = read_int();
                int d = read_int();
                int a = read_int();
                storea[number - 1] = a;
                stored[number - 1] = d;
            }
            show[cnt++] = output(storea, stored, n);
        }
    }
    for (int i = 0; i < cnt - 1; i++) printf("%d\n", show[i]);
    if (cnt > 0) printf("%d", show[cnt - 1]);
    free(attack);
    free(defense);
    free(stored);
    free(storea);
    free(show);
    return 0;
}
```

由于该题输入数据较多，在此补充一种不用 scanf 的数据读取方法，可以优化读取效率
