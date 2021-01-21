
#include <array>
#include <vector>
#include <stopwatch.h>
#include <iostream>

template <size_t size> std::array<int,size> randomArray();
void NormalMergeSort();
void cpuMergeSort(int* array, int leftIndex, int rightIndex);
void cpuMerge(int* array, int leftIndex,int midIndex, int rightIndex);

int main()
{
    NormalMergeSort();
    return 0;
}

void NormalMergeSort()
{
    constexpr int array_size = 25000;
    std::array<int, array_size> array = randomArray<array_size>();

    Stopwatch stopwatch;
    stopwatch.Start();
    cpuMergeSort(array.data(),0,array.size()-1);

    stopwatch.Stop();
    std::cout << "cpu merge sort " << stopwatch.Time() << std::endl;
}

void cpuMergeSort(int *array, int leftIndex, int rightIndex)
{
    if(leftIndex>=rightIndex)
    {
        return;//returns recursively
    }
    int midIndex = (leftIndex+rightIndex-1) / 2;
    cpuMergeSort(array,leftIndex,midIndex);
    cpuMergeSort(array,midIndex+1,rightIndex);
    cpuMerge(array,leftIndex,midIndex,rightIndex);
}

void cpuMerge(int *array, int leftIndex, int midIndex, int rightIndex)
{
    const int leftArraySize = midIndex - leftIndex + 1;
    const int rightArraySize = rightIndex - midIndex;

    //create temp vectors
    std::vector<int> L(leftArraySize);
    std::vector<int> R(rightArraySize);

    //copy data to temp arrays
    memcpy(&L[0], &array[leftIndex], leftArraySize * sizeof(int));
    memcpy(&R[0], &array[midIndex+1], rightArraySize * sizeof(int));

    //merge temp arrays back into array[l..r]
    int i =0, j = 0, k = leftIndex;

    while (i < leftArraySize && j < rightArraySize)
    {
        if (L[i] <= R[j])
        {
            array[k] = L[i];
            i++;
        }
        else {
            array[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of
    // L[], if there are any
    while (i < leftArraySize)
    {
        array[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of
    // R[], if there are any
    while (j < rightArraySize)
    {
        array[k] = R[j];
        j++;
        k++;
    }
}

template<size_t size>
std::array<int, size> randomArray() 
{
    std::array<int,size> array;

    for (int i = 0; i < size; ++i)
    {
        array[i] = rand() % 100000 + 1;
    }

    return array;
}
