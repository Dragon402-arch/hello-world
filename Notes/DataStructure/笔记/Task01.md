https://blog.csdn.net/qq_45525486/article/details/122928495

Task01数组
知识点总结
数组的定义
数组是具有一定顺序关系的若干对象组成的集合，组成数组的对象成为数组元素。
它可以存储一个 固定大小 的 相同类型 元素的顺序集合。
数组的元素也可以是数组，这样就得到了多维数组。

数组的存储
数组是由连续的内存位置组成，即在物理存储层面是使用一片连续的存储空间。
最低的地址对应第一个元素，最高的地址对应最后一个元素。 即按顺序连续存储。
这样的好处是我们可以利用索引对数组元素进行随机存取，而不足则是数组空间固定，不易扩充或减少，容易造成溢出或空间浪费。

数组元素的访问
数组的存储按照行优先（C、C++、C#）或列优先（Forturn）的原则进行。
可以利用以下公式计算得到数组中任一下标所存元素的位置，当然在C语言中我们可以直接利用下标对数组元素进行访问。（考研数据结构里面还是比较喜欢考下面这种计算的）
以下数组声明分别为 type a[n]; type a[m][n]; type a[m][n][l];

一维数组：Loc(a[i]) = Loc(a[0]) + i x c
二维数组：Loc(a[i][j]) = Loc(a[0][0]) + (i x n + j) x c
三维数组：Loc(a[i][j][k]) = Loc(a[0][0][0]) + (i x n x l + j x l + k) x c
以上公式中 c 即代表数组中单个元素大小，可以利用 sizeof(a[0]) 得到，也是数据类型type的大小

例题
Leetcode1 两数之和
这是一道很简单的题，很直接的思路就是暴力解法，利用二重循环穷举数组中两元素之和，找到满足的两元素返回下标即可，这样解答的时间复杂度是O(N2)；
此外，可以将数组先排序，再分别从头尾开始夹逼。
Leetcode的官方题解还提供了利用哈希表的解法，这样可以将时间复杂度降低至O(N)。
Leetcode16 最接近的三数之和
这是一道中等难度的题，也可以用暴力解法求解，即考虑直接使用三重循环枚举三元组，找出与目标值最接近的作为答案，但是时间复杂度将达到O(N3)；
Leetcode的官方题解中给出的解法是双指针法，即首先将数组元素升序排列，假设数组的长度为 n，我们需要的答案由 a、b、c 三个数得到，可以先枚举 a，它在数组中的位置为 i，然后在位置 [i+1,n) 的范围内利用双指针法枚举 b 和 c。具体为令指针 low 指向 i+1，即 b 的起始位置，另指针 high 指向 n-1，即 c 的起始位置；利用 a+b+c 与 target 比较的结果来更新答案，即若 a+b+c > target，就使low++，若a+b+c < target，就使high–，而当 a+b+c = target 时即可直接返回答案。
作业题
1 Leetcode27 移除元素
这道题看似很简单，但在代码实现时需要注意细节，且题目中还特别注明需要原地移除待删元素，即使用O(1)的额外空间。
对于数组我们没法像链表那样直接删除某个结点，只能用其他值来覆盖待删元素的原始值。同时为了让数组中剩余元素都集中在数组前部，即需要对数组元素进行移动，如果按照这样的思路，则每找到一个等于 val 的元素，都要将其后所有元素向前移动，即num[i]=num[i+1]，时间复杂度为O(N2)。
优化的算法是，利用双指针，让 i 指向当前将要处理的元素，j 指向下一个将要赋值的位置。如果 i 指向的元素不等于 val，则是需要保留的元素，就将 i 指向的元素复制到 j 指向的位置，然后将 i 和 j 同时右移；如果 i 指向的元素等于 val，则需要被覆盖，此时 i 不动，j 右移一位。当 i 遍历完整个数组后，j 的值即为结果数组的长度。
以上算法实现如下，另，以上算法还可以继续优化，即利用头尾双指针，具体可参考Leetcode官方题解。

int removeElement(int* nums, int numsSize, int val){
  int j=0;
  for(int i=0; i<numsSize; i++){
  	if(nums[i]!=val)
  		nums[j++]=nums[i];
  }
//	printf("%d\n",j);
//	for(int i=0; i<j; i++)
//		printf("%d ",nums[i]);
  return j;
}

2 Leetcode26 删除有序数组中的重复项
这道题与上一题极为相似，这里的重复项其实即为上一题中的val，又因为数组有序，所以重复项必定是连续的。利用双指针法，用 i 指向需保留的元素的存储位置，用 j 当前遍历元素的位置，当数组元素大于0时，数组中至少包含一个元素，因此 nums[0] 必定保留，所以从下标 1 开始删除重复元素。当 j 遍历到与保留元素不相等的元素时，即要保留，将其赋给 i，并将 i 和 j 向后移动；当 j 指向元素与 i 所指元素相等时，即为重复项，继续向后移动 j。
注意，这里的 i 指向的是结果数组中最后一个元素的位置，因此结果数组实际长度为 i+1 。
Leetcode官方题解中解法与此相同，但其具体实现时时用 nums[j] 与 nums[j-1] 相比。

int removeDuplicates(int* nums, int numsSize){
	int i=0;
	for(int j=1; j<numsSize; j++){
		if(nums[i]!=nums[j])
			nums[++i]=nums[j];
		}
	return i+1;
}

3 Leetcode15 三数之和
这道题与例题中的第2题也是大同小异，相信那道题看懂了的话，这道题的思路也会很透彻，我就不再赘述。
这里重点说一下用C语言的代码实现，如果是使用C++，那么丰富的STL可以很方便的操作和存储三元组，但是C语言则需要用到指针和内存管理相关的知识。
相信很多初学C语言的同学看到提交代码的函数头已是一头雾水，还好我不久前刚好看过这块儿相关的内容，还可以理解一部分，但是也有些不懂的地方，这里给大家推荐一下一本C语言学习手册。

回到这道题，首先需要明确的是，我们的结果是由若干三元组组成的一个数组，即一个二维数组，其中每个元素为含有三个元素的一维数组。再看这个函数头，
int** threeSum，该函数的返回值为结果二维数组的起始位置，即第一个三元组中第一个元素的位置；int* returnSize，表示一共有多少个三元组，需要在起始时赋值0，若不在第一行赋值便会报错，原因位置，还请知晓的大佬指教；int** returnColumnSizes，表示二维数组中每个元素（即一维数组）的大小，可见他也是一个二重指针，具体原因我也不知。

再看内存申请：

int** ret = (int**)malloc(sizeof(int*) * numsSize * numsSize);
ret[*returnSize] = (int*)malloc(sizeof(int) * 3);
1
2
以上两句分别是为作为结果的二维数组 ret 和 ret 中每新增的一个三元组 ret[*returnSize] 申请空间。
在C语言中数组名等同于起始地址，也就是说，数组名就是指向第一个成员的指针，所以 ret[] 就等同于* ret； 而 int** ret 就等同于int ret[][]，但是后者声明方法需要指出二维数组的大小，且声明后得到的内存空间将无法再进行改变，所以需要使用前者来动态申请一片空间。

*returnColumnSizes = (int*)malloc(sizeof(int) * numsSize * numsSize);
(*returnColumnSizes)[*returnSize] = 3;
1
2
至于这两句，我也着实存在疑问，不懂为什么要这样申请空间以及赋值，还请朋友们指点。

int cmp(const void* a, const void* b){
	return (*(int*)a - *(int*)b);
}
int** threeSum(int* nums, int numsSize, int* returnSize, int** returnColumnSizes){
    *returnSize = 0;
	if(numsSize < 3)
		return NULL;
	qsort(nums, numsSize, sizeof(int), cmp);
	int** ret = (int**)malloc(sizeof(int*) * numsSize * numsSize);
	*returnColumnSizes = (int*)malloc(sizeof(int) * numsSize * numsSize);
	for(int i=0; nums[i] <= 0 && i<numsSize-2; i++){
		if(i > 0 && nums[i]==nums[i-1])
			continue;
		int low = i+1, high = numsSize-1;
		while(low < high){
			int sum = nums[i] + nums[low] + nums[high];
			if(sum == 0){
				ret[*returnSize] = (int*)malloc(sizeof(int) * 3);
				ret[*returnSize][0] = nums[i];
				ret[*returnSize][1] = nums[low];
				ret[*returnSize][2] = nums[high];
				(*returnColumnSizes)[*returnSize] = 3;
				(*returnSize)++;
				while(low < high && nums[low] == nums[++low]);
				while(low < high && nums[high] == nums[--high]);
			}
			else if(sum < 0) low++;
			else high--;
		}
	}
//	printf("%d\n", *returnSize);
	return ret;
}
