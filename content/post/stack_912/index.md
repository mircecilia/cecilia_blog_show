---
date : '2025-03-16T22:47:53+08:00'
draft : flase
title : '链表栈应用'
categories:
  - Stack
  - DSAA
  - C
  - 912
---
## 代码实现

``` c
# include <stdio.h>
# include <stdlib.h>

typedef struct node{
    int data;
    struct node* next;
}node;

typedef struct stack{
    node* top;
}stack;

void initstack(stack* s, int n){
    for(int i = 0; i < n; i++){
        s[i].top = NULL;
    }
}

int isempty(stack* s,int x){
    return s[x-1].top == NULL;
}

void push(stack* s,int x,int data){
    node* newnode = (node*)malloc(sizeof(node));
    newnode->data = data;
    newnode->next = s[x-1].top;
    s[x-1].top = newnode;
}

int pop(stack* s,int x){
    if(isempty(s,x)){
        return 0;
    }
    node* temp = s[x-1].top;
    int tempvalue = temp->data;
    s[x-1].top = s[x-1].top->next;
    free(temp);
    return tempvalue;
}

int show(stack* s,int x){
    node* temp = s[x-1].top;
    int output = 0;
    while(temp != NULL){
        output += temp->data;
        temp = temp->next;
    }
    return output;
}


int main(){
    int n;
    scanf("%d",&n);
    stack s[n];
    initstack(s, n);
    int m;
    int cnt = 0;
    scanf("%d",&m);
    int store[m];
    while(m > 0){
        int symbol;
        scanf("%d",&symbol);
        if (symbol == 1){
            int x,w,c;
            scanf("%d %d %d",&x,&w,&c);
            while(c > 0){
                push(s,x,w);
                c--;
            }
            store[cnt] = show(s,x);
            cnt++;
        }
        else if(symbol == 2){
            int x,c;
            int temp = 0;
            scanf("%d %d",&x,&c);
            while(c > 0){
                temp += pop(s,x);
                c--;
            }
            store[cnt] = temp;
            cnt++;
        }
        else if(symbol == 3){
            int x,y;
            scanf("%d %d",&x,&y);
            while(!(isempty(s,x))){
                int temp = pop(s,x);
                push(s,y,temp);
            }
            store[cnt] = show(s,y);
            cnt++;
        }
        m--;
    }
    for(int i = 0;i < cnt - 1;i++){
        printf("%d\n",store[i]);
    }
    printf("%d",store[cnt-1]);
    return 0;
}
```

本题为链表栈的应用，思路无难度，但写代码时易错点很多，在此以 912 题目（2024）进行练习，以便后期复习
