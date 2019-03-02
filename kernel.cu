
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define DUMMY_VAL INT_MIN
#define MAX_NUM_OF_BLOCKS 1024
#define NUM_OF_THREADS_PER_BLOCK 512
#define INIT_MAX_THREAD_ARR_SIZE 4 

struct Node
{
	int val;
	Node *next;
	Node *prev;
} typedef Node_t;

cudaError_t FastQuickSortByCuda(int *arr, int size);

__device__ void trasnferArrToAllocatedCircleList(int *arr, Node_t *list, int arrSize)
{
	// built a dummy start
	Node *dummy = list;
	dummy->val = DUMMY_VAL;
	dummy->next = list + 1;
	dummy->prev = list + arrSize;
	list[1].prev = dummy;
	list[arrSize].next = dummy;

	int i;
	for (i = 1; i <= arrSize; i++)
	{
		list[i].next = list + ((i + 1) % (arrSize + 1));
		list[i].prev = list + (i - 1);
		list[i].val = arr[i - 1];
	}
}

__device__ void swapVals(Node_t *a, Node_t *b)
{
	int t = a->val;
	a->val = b->val;
	b->val = t;
}

// return a pointer to a node which devided the list for 2 parts: low or equal to 'partitionVal', and greater tha it.
// The node is the last one in list, which eqaul or less than 'partitionVal'
// Example: partitionVal = 5, {1, 4, 2, 3, 8, 6, 12} => the node is the one with val 3
__device__ Node_t* partition_AsCircleList(Node_t* l_node, Node_t* h_node, int partitionVal)
{
	int p = partitionVal;

	Node_t *i_node = l_node->prev; //(l - 1);

	Node_t* j_node;
	for (j_node = l_node; j_node != h_node; j_node = j_node->next)
	{
		if (j_node->val <= p)
		{
			i_node = i_node->next; //i++;
			swapVals(i_node, j_node); //swap(&arr[i], &arr[j]);
		}
	}

	if (j_node->val <= p)
	{
		i_node = i_node->next;
		swapVals(i_node, h_node); //swap(&arr[i + 1], &arr[h]);
	}

	return i_node;
}


__device__ void moveSubListAfter_CircleList(Node *start, Node *end, Node *after)
{
	Node *prev_start = start->prev;
	Node *next_end = end->next;

	// connect the nodes before and after the subList we move
	prev_start->next = next_end;
	next_end->prev = prev_start;

	Node *next_after = after->next;

	after->next = start;
	start->prev = after;
	next_after->prev = end;
	end->next = next_after;
}

__device__ void connectBetween(Node *from, Node *to)
{
	from->next = to;
	to->prev = from;
}

__device__ void putDummyBetween(Node *from, Node *to, Node *dummy)
{
	from->next = dummy;
	dummy->prev = from;

	to->prev = dummy;
	dummy->next = to;
}

__device__ void swapSubLists_CircleList(Node_t *start_A, Node_t *end_A, Node_t *start_B, Node_t *end_B)
{
	Node_t *Prev_A = start_A->prev;
	Node_t *Next_A = end_A->next;
	Node_t *Prev_B = start_B->prev;
	Node_t *Next_B = end_B->next;
	int areAandBclosed = 0;
	int areBandAclosed = 0;
	Node_t dummy_A_To_B;
	Node_t dummy_B_To_A;

	if (end_A == Prev_B && start_B == Next_A)  // A --> B 
	{
		areAandBclosed = 1;

		putDummyBetween(end_A, start_B, &dummy_A_To_B);

		Next_A = end_A->next;
		Prev_B = start_B->prev;
	}

	if (end_B == Prev_A && start_A == Next_B)  // B --> A 
	{
		areBandAclosed = 1;

		putDummyBetween(end_B, start_A, &dummy_B_To_A);

		Prev_A = start_A->prev;
		Next_B = end_B->next;
	}

	// algorithm:
	moveSubListAfter_CircleList(start_A, end_A, Prev_B);

	moveSubListAfter_CircleList(start_B, end_B, Prev_A);

	if (areAandBclosed == 1)  // B --> dummy --> A
		connectBetween(end_B, start_A);

	if (areBandAclosed == 1)  // A --> dummy --> B
		connectBetween(end_A, start_B);
}

__device__ void swapSubLists_CircleList_NotIncludeEnds(Node_t *start_A, Node_t *end_A, Node_t *start_B, Node_t *end_B)
{
	Node *end_A_prev = end_A->prev;
	Node *end_B_prev = end_B->prev;

	if (start_A == end_A && start_B == end_B)
		return;
	else if (start_A == end_A)
		moveSubListAfter_CircleList(start_B, end_B_prev, end_A);
	else if (start_B == end_B)
		moveSubListAfter_CircleList(start_A, end_A_prev, end_B);
	else
		swapSubLists_CircleList(start_A, end_A_prev, start_B, end_B_prev);
}

__device__ int getCircleListSize(Node* list)
{
	Node *currentNode = list->next;
	int i = 0;
	while (currentNode != list)
	{
		i++;
		currentNode = currentNode->next;
	}
	return i;
}


__device__ void bubbleSort_CircleList(Node_t *list)
{
	Node_t *i = list->next, *j = i->next;
	while (i != list)
	{
		while (j != list)
		{
			if (i->val > j->val)  // if the prev number bigger than a next node
				swapVals(i, j);
			j = j->next;
		}
		i = i->next; // i++
		j = i->next; // j = i + 1
	}

}

__device__ void backFromListToArr(int *arr, Node **dummyPointers, int num_of_threads)
{
	int currentOffset = 0;
	int id = 0;

	for (id = 0; id < num_of_threads; id++)
	{
		Node_t *currentDummy = dummyPointers[id];
		Node_t *node = currentDummy->next;
		while (node != currentDummy)
		{
			arr[currentOffset++] = node->val;
			node = node->next;
		}
	}
}

/*  Original Source: https://www.uio.no/studier/emner/matnat/ifi/INF3380/v10/undervisningsmateriale/inf3380-week12.pdf
	
	The big idea: there are 2^M threads and array in size N. Each thread has it's own part from the array.
				  Which means - each thread responsible on (N / 2^M) numbers (in average)
				  Steps of an iteration:
					1) We choose a pivot val A for 2^K threads. 
					2) We devide the threads for 2 eqaul groups:
						0: threads with id of 0 til (2^(M-1))-1 (include)
						1: threads with id of 2^(M-1) til (2^M)-1 (include)
					3) Group 0 gives the numbers which greater than A to group 1,
						and take from group 1 the numbers which are lower of eqaul to A.
					4) Return to step 1 for each group: 0 and 1.  

	In theory: if we have N/2 threads, the algorithm tends to runtime of O(log(N)) in avg case.
				The "drawback" of this algorithm is that the 'risk' to get the worst case: O(N^2) not changed.
				My opinion: by good choise of a pivot value, we can reduce that risk.
							The idea: each thread (in each iteration) choose the number from its 
									  part of the array with the lowest variance (O(1)) 
									  and each thread even thread compares with the odd thread O(log(N)) 
							the problem: the avg case becomes O(log(N)*log(N))
	
	In Code: there is transformation from an array to double-linked list.
			the code transforms the list back to array in runtime of O(N).
	
	Conclusion: Although the code does not implement the theory, if N = 2^25, this code is 25 faster than the 
				original quick sort.
*/
__device__ 
void PerformQuickSort(int *arr, int size_arr, Node_t *bigList, Node_t **dummyPointers, Node_t **pivotPointers
	, int num_of_threads)
{
	// assumption: bigList size = size_arr + num_of_threads
	int myid = blockIdx.x * blockDim.x + threadIdx.x;

	// Alert: myArrOffset going to the dummy of the thread
	// Alert: 'myArrSize' first definition is wrong for the last thread,
	//			the 'if' statement right it
	int myArrSize = (size_arr + num_of_threads - 1) / num_of_threads;
	int myArrOffset = (myid * myArrSize);  // works for the last thread
	int myListOffset = myArrOffset + myid;  // = myid * (myArrSize + 1). the '+ myid' is for the dummies

	if (myid == num_of_threads - 1) // if it's the last thread
		myArrSize = size_arr - ((num_of_threads - 1) * myArrSize);

	Node_t* myList = bigList + myListOffset;
	dummyPointers[myid] = myList;

	trasnferArrToAllocatedCircleList(arr + myArrOffset, myList, myArrSize);


	int i;
	int n;  // num of threads each group
	int pivotPointers_index;
	int myPivotVal;
	for (i = num_of_threads; i >= 2; i = i / 2)  // assumption: num_of_threads = 2^m
	{
		pivotPointers[myid]->val = -1;
		// example: if num of threads are 32, so there are 16 threads each group
		// in the next iteration there will be 4 groups (instead of two), which means 8 threads each group
		n = i / 2;

		pivotPointers_index = myid / i;
		if (myid == num_of_threads - 1)
		{
			printf("\nThread id = %d", myid);
			printf("\npivotPointers_index = %d", pivotPointers_index);
		}

		//int idForWaitingFor = myid;

		if (myid % i == 0)  // the thread/s who provide the pivot
		{
			// choose pivot index
			// assumption: myList is not empty (in over word: the prev of dummy is not the dummy itself)

			pivotPointers[pivotPointers_index] = myList->prev;  // the last node in my list  
															// there is a better way to choose a pivot? 
		}
		// wait until the 'if' statement provides
		__syncthreads();

		// Continue of the example:
		// i = 32, so each thread from 0 til 31 has the same 'myPivotVal'
		myPivotVal = pivotPointers[pivotPointers_index]->val;

		
		pivotPointers[myid] = partition_AsCircleList(myList->next, myList->prev, myPivotVal);  // the heart of quicksort
		// pivotPointers contains the pointers for the last node that devide 'myList' to two subLists:
		// myDummy --> pivotPointers[myid] : all nodes that contains less or equal to 'myPivotVal'
		// pivotPointers[myid]->next --> myDummy: all nodes that contains greater than 'myPivotVal

		__syncthreads();

		if ((myid / n) % 2 == 0)  // if my group is even
		{
			// TODO: swap of parts of the list in each myid + n
			// example: 16 threads -> i = 16 -> n = 8.
			// we want thread 0 will "talk" with thread 8
			// thread 1 will "talk" with thread 9
			// and so on until 7 and 15
			int otherId = myid + n;

			// continue: we take a senario - thread 0 "talk" to thread 8
			// thread0 = {3, 1, 2, 7, 8, 9, 5}
			// thread8 = {6, 4, 0};
			// lets say that: pivotVal = 5
			// after pivotting we got: thread0 = {3, 1, 2, 5, 7, 8, 9}    thread8 = {4, 0, 6};
			// pivotThread0 = 5, pivotThread8 = 0
			// we want that thread 0 will have all values that <= 5 (pivotVal), and the rest in thread 8
			// in other word: to swap between sublist {7, 8, 9} and {4, 0}
			// such that, we get: thread 0 contains all "low" nodes, and thread 8 contains the "high" nodes
			swapSubLists_CircleList_NotIncludeEnds
				(pivotPointers[myid]->next, dummyPointers[myid],
					dummyPointers[otherId]->next, pivotPointers[otherId]->next);
		}

		__syncthreads();
	}

	bubbleSort_CircleList(myList);

	__syncthreads();

	// copy the values from all lists to arr
	// O(arr + num_of_threads) - can make it better?
	if (myid == 0)
		backFromListToArr(arr, dummyPointers, num_of_threads);
}

/*
Steps:
1) each thread calc his: myid, myArrSize, myArrOffset
*/
// Sizes:  bigList = N + M, dummyPointers = M, pivotPointers = M
__global__ void fastQuickSortKernel(int *arr, int size_arr, Node_t *bigList, Node_t **dummyPointers, Node_t **pivotPointers, int num_of_threads)
{
	int myid = blockIdx.x * blockDim.x + threadIdx.x;

	if (myid < num_of_threads)
		PerformQuickSort(arr, size_arr, bigList, dummyPointers, pivotPointers, num_of_threads);
}


void createRandomArray(int *arr, int size)
{
	int i;
	for (i = 0; i < size; i++)
		arr[i] = rand() % size;
}

int assertSorted(int *arr, int size)
{
	int i;
	for (i = 0; i < size - 1; i++)
		if (arr[i] > arr[i + 1])
			return 0;
	return 1;
}

int main()
{
	// TODO: srand(time(NULL));   // Random Initialization, should only be called once.
	const int size = 10000;
	int *rands = (int*)calloc(size, sizeof(int));
	createRandomArray(rands, size);

	// sort
	cudaError_t cudaStatus = cudaSuccess;
	cudaStatus = FastQuickSortByCuda(rands, size);

	// check if there was not an error
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	// check if the array really sorted

	if (!assertSorted(rands, size))
	{
		fprintf(stderr, "array not sorted!");
		return 1;
	}

	return 0;
}


/*
Steps:
1. Choose GPU
2. Allocation for GPU:
- the array (N ints)
- lists of N nodes (1 node for each number) -> C*N [ints bytes] <= 2N [ints bytes]
- addresses's array for the list (for each proccess there are 2 adresses:
1 for the start of the list' proccess, and 1 for the pivot)
assumption: there will be no more than N/2 threads, so: memory <= N [ints bytes]

*/


cudaError_t FastQuickSortByCuda(int *arr, int size, int num_of_blocks, int numActivatedThreads)
{
	// Simulation:

	cudaError_t cudaStatus;

	// Sizes:  bigList = N + M, dummyPointers = M, pivotPointers = M

	int *dev_arr;
	Node_t *dev_list;
	Node_t **dev_dummyPointers; // addresses
	Node_t **dev_pivotPointers;

	int size_arr = size;
	int size_list = size_arr + numActivatedThreads;
	int size_dummyPointers = numActivatedThreads;
	int size_pivotPointers = numActivatedThreads;


	// 1. Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// 2. Allocate GPU buffers for array, list and helper array
	// arr:
	cudaStatus = cudaMalloc((void**)&dev_arr, size_arr * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "arr: cudaMalloc failed!");
		goto Error;
	}

	// list:
	cudaStatus = cudaMalloc((void**)&dev_list, size_list * sizeof(Node_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "list: cudaMalloc failed!");
		goto Error;
	}

	// dummy' pointers array:
	cudaStatus = cudaMalloc((void**)&dev_dummyPointers, size_dummyPointers * sizeof(Node_t*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dummyPointers: cudaMalloc failed!");
		goto Error;
	}

	// pivot' pointers array:
	cudaStatus = cudaMalloc((void**)&dev_pivotPointers, size_pivotPointers * sizeof(Node_t*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "pivots: cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_arr, arr, size_arr * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// TODO: cakc num of blocks
	// Launch a kernel on the GPU with one thread for each element.
	// TODO: change from 1 to num of blocks
	fastQuickSortKernel << <num_of_blocks, NUM_OF_THREADS_PER_BLOCK >> >(dev_arr, size_arr, dev_list, dev_dummyPointers, dev_pivotPointers, numActivatedThreads);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "QuickSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(arr, dev_arr, size_arr * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	// TODO: free
	cudaStatus = cudaFree(dev_arr);
	cudaStatus = cudaFree(dev_list);
	cudaStatus = cudaFree(dev_dummyPointers);
	cudaStatus = cudaFree(dev_pivotPointers);

	return cudaStatus;
}

cudaError_t FastQuickSortByCuda(int *arr, int size_arr, int maxThreadArrSize)
{
	int num_of_threads_activated, num_of_threads_power_2;
	int num_of_blocks;
	do
	{
		num_of_threads_activated = (size_arr + (maxThreadArrSize - 1)) / maxThreadArrSize;
		num_of_threads_power_2 = 1; // 2 power 0
		while (num_of_threads_power_2 < num_of_threads_activated)
			num_of_threads_power_2 *= 2;

		num_of_threads_activated = num_of_threads_power_2; // example: 2^13
		num_of_blocks = (num_of_threads_activated + (NUM_OF_THREADS_PER_BLOCK - 1)) / NUM_OF_THREADS_PER_BLOCK;
		maxThreadArrSize += INIT_MAX_THREAD_ARR_SIZE;
	} while (num_of_blocks > MAX_NUM_OF_BLOCKS);

	cudaError_t status = FastQuickSortByCuda(arr, size_arr, num_of_blocks, num_of_threads_activated);
	return status;
}

cudaError_t FastQuickSortByCuda(int *arr, int size_arr)
{
	cudaError_t status = FastQuickSortByCuda(arr, size_arr, INIT_MAX_THREAD_ARR_SIZE);
	return status;
}
