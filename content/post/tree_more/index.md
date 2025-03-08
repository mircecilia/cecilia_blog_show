---
date : '2025-03-06T16:06:38+08:00'
draft : flase
title : '二叉树进阶'
categories:
  - tree
  - DSAA
  - C
---

## BST

- 结点定义

```c
typedef struct node{
    int value;
    struct node* left;
    struct node* right;
} node;
```

- 用值创造结点

``` c
node* creatNode(int value){
    node* newnode = (node*)malloc(sizeof(node));
    newnode->value = value;
    newnode->left = NULL;
    newnode->right = NULL;
    return newnode;
}
```

- 在已有树的基础上插入新结点

``` c
node* insert(node* root, int value){
    if(root == NULL){
        return creatNode(value);
    }
    if(value<root->value){
        root->left = insert(root->left,value);
    }
    else if(value>root->value){
        root->right = insert(root->right,value);
    }
    return root;
}
```

- 搜索结点

```c
node* search(node* root,int value){
    if(root->value == value || root == NULL){
        return root;
    }
    if(value<root->value){
        return search(root->left,value);
    }
    return search(root->right,value);
}
```

- 寻找输入树中的最小结点

```c
node* findmin(node* root){
    node* current = root;
    while(current != NULL && current->left != NULL){
        current = current->left;
    }
    return current;
}
```

- 删除结点

``` c
node* delete(node* root, int value){
    if(root == NULL){
        return root;
    }
    if(value < root->value){
        root->left=delete(root->left,value);
    }
    else if(value > root->value){
        root->right=delete(root->right,value);
    }  // 前半部分寻找结点
    else if(value == root->value){ // 找到结点后
        if(root->left == NULL){ // 若删除的结点没有或只有一个结点
            node* temp = root->right;
            free(root);
            return temp;
        }
        else if(root->right == NULL){
            node* temp = root->left;
            free(root);
            return temp;
        }
        // 若结点有两个结点，则找到其右树的最小结点将其代替
        node* temp = findmin(root->right);
        root->value = temp->value;
        root->right = delete(root->right,temp->value);
    }
    return root;
}
```

- 删除树

``` c
void freeTree(node* root) {
    if (root != NULL) {
        freeTree(root->left);
        freeTree(root->right);
        free(root);
    }
}
```

## AVL

- 定义结点

``` c
typedef struct node{
    int data;
    struct node* left;
    struct node* right;
    int height;
} node;
```

- 获取高度及平衡因子

``` c
int getheight(node* root){
    if(root == NULL){
        return 0;
    }
    return root->height;
}

int getbalance(node* root){
    if(root == NULL){
        return 0;
    }
    return getheight(root->left) - getheight(root->right);
}
```

- 用值创造新的结点

``` c
node* creatnode(int data){
    node* root = (node*)malloc(sizeof(node));
    root->data = data;
    root->left = NULL;
    root->right = NULL;
    root->height = 1;
    return root; 
}
```

- 左旋及右旋

``` c
node* leftrotate(node* root){
    node* r = root->right;
    node* l = r->left;
    r->left = root;
    root->right = l;
    root->height = max(getheight(root->left),getheight(root->right))+1;
    r->height = max(getheight(r->left),getheight(r->right))+1;
    return r;
}

node* rightrotate(node* root){
    node* l = root->left;
    node* r = l->right;
    l->right = root;
    root->left = r;
    root->height = max(getheight(root->left) , getheight(root->right))+1;
    l->height = max(getheight(l->left) , getheight(l->right))+1;
    return l;
}
```

在此详细解释两种操作：  
左旋操作为储存操作结点root的右结点r，以及右结点r的左结点l，将原root结点下降至原l结点的位置，并将储存的原l结点变为现l结点（即原root结点）的右结点  
右旋操作为储存操作结点root的左结点l，以及左节点l的右结点r，将原root结点下降至原r结点的位置，并将储存的原r结点变为现r结点（即原root结点）的左结点  
两种操作的本质内核都是尽可能降低原树的不平衡度，使原树中没有结点的位置被最大程度利用，从而使得树更加平衡

- 插入结点

``` c
node* insert(node* root, int data){
    if(root==NULL){
        return creatnode(data);
    }
    if(data < root->data){
        root->left = insert(root->left,data);
    }
    else if(data > root->data){
        root->right = insert(root->right,data);
    } // 寻找结点应插入的位置
    else if(data == root->data){
        return root; // 插入相同结点则返回原有结点
    }
    // 找到插入位置
    root->height = max(getheight(root->left),getheight(root->right))+1; // 实时更新树的高度

    int balance = getbalance(root); // 计算目前的平衡因子
    // 四种情况分类
    if(balance > 1 && getbalance(root->left) >= 0){
        return rightrotate(root);
    }
    if(balance < -1 && getbalance(root->right) <= 0){
        return leftrotate(root);
    }
    if(balance > 1 && getbalance(root->left) < 0){
        root->left = leftrotate(root->left);
        return rightrotate(root);
    }
    if(balance < -1 && getbalance(root->right) > 0){
        root->right = rightrotate(root->right);
        return leftrotate(root);
    }
    return root;
}
```

四种分类情况分别代表了不同的失衡类型，若root结点的左树失衡，则第一类型为L，若右树失衡则第一类型为R，再进一步观察失衡树的平衡因子以确定下一步进行左旋还是右旋，左树高则为L型，右树高则为R型  
LL型为左树严重失衡，使用右旋  
RR型为右树严重失衡，使用左旋  
LR型则先左旋左结点变为LL型，再右旋  
RL型则先右旋右结点变为RR型，再左旋  
总结来说，旋转操作的左右旋与类型的R与L相反,按步骤反向进行即可

- 删除结点

``` c
node* delete(node* root , int data){
    if(root == NULL){
        return root;
    }
    if(data < root->data){
        root->left = delete(root->left , data);
    }
    else if(data > root->data){
        root->right = delete(root->right,data);
    } // 寻找要删除的结点
    else if(data == root->data){
        if(root->left == NULL){
            node* temp = root->right;
            free(root);
            return temp;
        }
        else if(root->right == NULL){
            node* temp = root->left;
            free(root);
            return temp;
        } // 删除结点若没有或只有一个子结点，则直接将子结点返回
        else{
            node* temp = findmin(root->right);
            root->data = temp->data;
            root->right = delete(root->right,temp->data);
        } 
        // 若含有两个结点，则将其右子树的最小结点替换自身，并删除右子树的最小结点
    }
    
    if (root == NULL) {  // 添加检查
        return root;
    }

    root->height = max(getheight(root->left),getheight(root->right)) + 1; // 更新树的高度

    int balance = getbalance(root); // 更新平衡因子

    if (balance > 1 && getbalance(root->left) >= 0){
            return rightrotate(root);
    }
    if (balance > 1 && getbalance(root->left) < 0){
        root->left = leftrotate(root->left);
        return rightrotate(root);
    }
    if(balance < -1 && getbalance(root->right) <= 0){
        return leftrotate(root);
    }
    if(balance < -1 && getbalance(root->right) > 0){
        root->right = rightrotate(root->right);
        return leftrotate(root);
    } // 此处四种情况同插入，本质是使新树平衡度提高，提升空间利用率
    return root;
}
```

## 总结

AVL的构建要求高于BST，更加严格  
因此AVL可以防止插入数据顺序过于整齐而导致整个树退化为链表，从而使时间复杂度由 log(n) 提升至 n  
AVL的插入，查找，删除时间复杂度均为 log(n)，但若进行频繁的插入，删除，红黑树的性能会优于AVL,因此 AVL 主要用于查找
