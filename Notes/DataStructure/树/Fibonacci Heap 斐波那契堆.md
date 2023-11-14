# Fibonacci Heap 斐波那契堆

Like Binomial Heap, Fibonacci Heap is a collection of trees with min-heap or max-heap property. In Fibonacci Heap, trees can have any shape even all trees can be single nodes (This is unlike Binomial Heap where every tree has to be Binomial Tree). 

在二项堆中用0阶二项树表示单个节点，而在斐波那契堆中则直接使用节点来表示。斐波那契堆的实例如下图所示。

![FibonacciHeap](https://media.geeksforgeeks.org/wp-content/uploads/Fibonacci-Heap.png)

 All tree roots are connected using circular doubly linked list.所有的根节点之间使用双向循环链表进行连接。

![img](https://www.tutorialspoint.com/assets/questions/media/41061/fibonacci_heap.jpg)

A Fibonacci heap can contain many trees of min-ordered heap. The roots of these trees are doubly linked and form a circular loop as in the case with siblings. The number of trees of a Fibonacci heap is always the number of roots.

