# Task02 链表

https://blog.csdn.net/qq_45525486/article/details/123010175

## 知识点总结

1. 单链表
   单链表是一种链式存取的数据结构，用一组地址任意的存储单元存放线性表中的数据元素。**即逻辑上位置相邻的元素其物理位置不一定相邻。**
   链表中的数据是以结点来表示的，每个结点的构成：元素(数据元素的值) + 指针(后继元素的存储位置)。

```c
struct ListNode {int val;struct ListNode *next;
}; //C语言中利用结构体定义一个简单的单链表
```

1. 头结点
   单链表分为**带头结点**和**不带头结点**两种，使用比较多的是**带头结点**的单链表，因为这样可以使得对于链表中的每个结点的操作都是统一的，增删改查都更方便，而**不带头结点**的单链表在进行一些操作时要区分第一个结点与其他结点，在编码实现上会造成不便。*但是Leetcode中好像多是用的不带头结点的单链表，这是我们常需要自己定义一个头结点。*
   **注意：** 任一链表不论是否带头结点，都是有头指针的，这是链表的起点，带头结点的单链表的头指针指向头结点，不带头指针的单链表的头指针指向第一个结点。
2. 单链表的插入、删除、创建（以带头结点的为例）
   插入：temp = cur->next; cur->next = a; a->next = temp; (在当前元素cur之后查入元素a)
   删除：cur->next == cur->next->next; (删除当前元素cur的后继元素)
   创建：单链表的创建有头插法和尾插法两种。**尾插法**顾名思义，就是把新元素一个一个插入到链表的末尾，即让cur->next = a; 而**头插法**则是将链表接在新元素的后面，也就是将新元素一个一个插入到链表头部，即a->next = cur; *通常可以利用头插法来完成链表的翻转。*
3. 衍生
   由于单链表只能按顺序逐个向后遍历，不能像数组一样随机存取，所以效率较低，为了改善这一点，又衍生出了双链表（每一个结点还用一个指向前序结点的指针）、循环链表（链表的最后一个元素的next指针指向头结点）和魂环双链表（双链表与循环链表结合）。

## 例题

### 1 Leetcode203 移除链表元素

**题目链接：** https://leetcode-cn.com/problems/remove-linked-list-elements/
**解题思路：** 这道题的思路很清晰，顺序的遍历链表，当遍历到与val值相等的结点时将其删除即可。这里需要注意的是，头结点head有可能被删除，所以定义新的头结点newhead指向head，用指针cur指向newhead，用cur->next来遍历链表，当cur->next->val == val时，删除cur->next。**因为如果直接用cur来遍历的话，cur所指元素需要被删除时，就需要知道他的前序结点，所以会带来不便。**
另外还可以用**递归**的方法求解，即先对除了头节点 head 以外的节点进行删除操作，然后再判断 head 的节点值是否等于给定的 val，若相等则新的头结点为head->next。在每次递归中将 head->next 作为一个新链表来输入，其终止条件是 head == NULL。可以参考官方题解。
**C语言实现代码：**

```c
struct ListNode* removeElements(struct ListNode* head, int val){if(!head)return head;// 新建头结点，连接原头结点struct ListNode* newhead = (struct ListNode*)malloc(sizeof(struct ListNode));newhead->next = head;struct ListNode* cur = newhead;// 用cur->next遍历链表while(cur->next){// 当cur->next的值等于val时，删除cur->nextif(cur->next->val == val)cur->next = cur->next->next;else cur = cur->next;}// 返回新头结点的后继return newhead->next;
}
```

### 2 Leetcode61 旋转链表

**题目链接：** https://leetcode-cn.com/problems/rotate-list/
**解题思路：** 这道题挺好，向右移动链表结点，实质上是头结点和尾结点的变化，另外为了方便实现，需要将链表头尾相接。通过仔细观察所给的示例，就可以发现其中的规律，即当 k 小于链表长度 len 时，移动 k 个位置后，新尾结点在原尾结点前 k 个位置，即倒数第 len-k 个。**注意**，当 k 为 len 的整数倍时，链表其实未移动，当 k > len 时，链表实际移动位置为 k - n x len，所以需要让 k 对 len 取模。**其他细节详见以下代码，第一次完全靠自己写出用时击败100%的代码还是很开心的，哈哈O(∩_∩)O。**
**C语言代码实现：**

```c
struct ListNode* rotateRight(struct ListNode* head, int k){// 当链表为空或只有一个元素，或移动次数为0时，直接返回if(!head || !head->next || k==0)return head;// 用len来记录链表的长度int len = 1;struct ListNode* cur = head;while(cur->next){len++;cur = cur->next;}// 若k是len的整数倍，则链表实际未移动位置，直接返回if(k % len == 0)return head;// 将原来的尾结点与头结点相接，使其成环cur->next = head;// 由于k可能大于len，所以让k对len取模k = k % len;// 移动后新的尾结点即为的原来的倒数第len-k个结点k = len - k;while(k--)cur = cur->next;// 找到新的尾结点，其后即为新的头结点head = cur->next;// 尾结点的next指针置空cur->next = NULL;return head;
}
```

## 作业题

### 1 Leetcode21 合并两个有序链表

**题目链接：** https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/
**解题思路：** 这是比较经典的一个算法，在排序算法中的**归并排序**中就要用到这一算法。思路也很清晰，即同时遍历两个链表， 比较两列表当前遍历到的结点，哪个结点的值小，就把该结点复制到新链表上就可以了，然后继续向后移动该指针，与之前大是那个相比较，以此类推，直到有一个链表已经遍历结束，则把另一链表的剩余部分直接接在新链表之后即可。时间复杂度为O(m+n)，m和n为两链表长度。另外，官方题解还提供了一种递归的方法，感兴趣的朋友可以参考官方题解。
**C语言代码实现：**

```c
struct ListNode* mergeTwoLists(struct ListNode* list1, struct ListNode* list2){// 若有某个链表为空则直接返回另一个if(list1 == NULL) return list2;if(list2 == NULL) return list1;struct ListNode* l1 = list1;struct ListNode* l2 = list2;// 新建一个带头结点的链表用来存放合并后的新链表struct ListNode* head = (struct ListNode*)malloc(sizeof(struct ListNode));head->next = NULL;struct ListNode* ret = head;// ret指针指向结果链表的尾结点while(l1 != NULL && l2 != NULL){// 利用指针l1和l2分别遍历比较两个输入链表的各元素// 将其中较小者复制到结果链表尾结点之后if(l1->val > l2->val) {ret->next = l2;l2 = l2->next;}else{ret->next = l1;l1 = l1->next;}ret = ret->next;// 每插入一个新元素，将ret向后移动}//在list1和list2中一者已经遍历完后，将另一者的剩余部分直接赋给结果链表ret->next = (l1 == NULL) ? l2 : l1;// 返回结果时需返回头结点的后继结点的位置return head->next;
}
```

### 2 Leetcode160 相交链表

**题目链接：** https://leetcode-cn.com/problems/intersection-of-two-linked-lists/
**解题思路：** 这道题也是比较经典，但是解题方法不好想，当然有些朋友直接的想到了用哈希表，确实是可行的。而更好的方法则是通过重复遍历两个链表来寻找交点。因为假设listA长度为a，listB长度为b，若两链表相交于一个结点C，这个C之后的结点也必然是listA和listB的公共结点，设包括C在内共有c个公共结点，那么就有**a+(b-c) = b+(a-c)**，即遍历一遍listA再遍历listB的前半部分和遍历一遍listB再遍历listA的前半部分，所需用的时间相同，他们会同时到达结点C；若不相交，则有**a+b = b+a**，即他们同时到达NULL。因此可以同时开始遍历两链表，当遍历结束时开始从头遍历另一链表，直到while结束。时间复杂度为O(m+n)，m和n为两链表长度。
**C语言代码实现：**

```c
struct ListNode *getIntersectionNode(struct ListNode *headA, struct ListNode *headB) {struct ListNode *a = headA;struct ListNode *b = headB;// 若链表A与B有交点，则必定会在交点处跳出while；// 若没有交点，则必定在a和b都等于NULL时跳出whilewhile(a != b){a = a->next;b = b->next;// 若链表A已经遍历完，则开始遍历链表Bif(a == NULL && b != NULL)a = headB;// 若链表B已经遍历完，则开始遍历练笔Aif(b == NULL && a != NULL)b = headA;}return a;
}
```

### 3 Leetcode82 删除排序链表中的重复元素 II

**题目链接：** https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/
**解题思路：** 这道题还有有一些难度的，我自己想利用快慢指针的方法来实现，当检测当slow和fast所指结点值相等时，则将fast一直移动到与slow不相等的结点处，然后将slow的前序结点的next指向fast，就完成了重复结点段的删除，思路很清晰，但实现起来却有很多坑，首先要注意保留slow的前序结点，另外在删除一段后要更新slow和fast的位置，最主要的还要slow和fast的判空，恕我愚笨，自己写了一个多小时，也无法通过全部用例，所以参考了别人的题解。
以下代码实现的是官方题解中的一种办法，简单易懂，实现方便，但也要注意，因为输入链表的第一个结点有可能要被删除，所以需要定义一个新的头结点。在遍历链表时，如果cur->next 与 cur->next->next 对应的结点值相等，就将 cur->next 的值记为 x，然后将其及后面所有值等于 x 的链表结点删除，直到cur.next 为空节点或者其元素值不等于 x 为止；如果当前 cur->next 与 cur->next->next 对应的元素不相同，那么说明 cur->next不重复，就可以将 cur 指向 cur->next。

**C语言代码实现：**

```c
struct ListNode* deleteDuplicates(struct ListNode* head){// 若链表为空或只有一个元素，则直接返回if(!head || !head->next)return head;// 为输入链表创建一个头结点，并用指针cur指向头结点struct ListNode* newhead = (struct ListNode*)malloc(sizeof(struct ListNode));newhead->next = head;struct ListNode* cur = newhead;while(cur->next && cur->next->next){if(cur->next->val == cur->next->next->val){// 当cur后连续的两个结点值相等时，说明出现了重复结点，记录该值int x = cur->next->val;// 将cur之后所有值为x的结点删除while(cur->next && cur->next->val == x)cur->next = cur->next->next;}else cur = cur->next;// 若cur后连续的两个结点值不等，则说明cur->next是不重复的，将其保留}// 返回头结点的后继即可return newhead->next;
}
```