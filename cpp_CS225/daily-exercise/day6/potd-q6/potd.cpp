// your code here!
#include "potd.h"
#include <iostream>
#include <cmath>

int *potd::raise(int *arr)
{
    int size = 1;
    for (int i = 0; arr[i] != -1; i++)
    {
        size += 1;
        // std::cout << size << " " << std::endl;
    }
    // std::cout << "Array size: " << size << " " << std::endl;
    int *arr_ptr_heap = new int[size];
    for (int i = 0; i < size; i++)
    {
        // std::cout << "Input: " << arr[i] << " i: " << i << " " << std::endl;

        if (i < size - 2)
        {
            arr_ptr_heap[i] = pow(arr[i], arr[i + 1]);
        }
        else
        {
            arr_ptr_heap[i] = arr[i];
        }

        // std::cout << arr_ptr_heap[i] << " " << std::endl;
    }
    return arr_ptr_heap;
}