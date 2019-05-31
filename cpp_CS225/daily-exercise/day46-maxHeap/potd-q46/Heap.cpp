
#include "Heap.h"
#include <iostream>
#include <algorithm>
void Heap::_percolateDown(int hole)
{
    // your code here
    int boundary = _data.size() - 1;
    int head = _data[1];
    for (; hole < boundary && 2 * hole <= boundary && head <= std::max(_data[hole * 2], _data[hole * 2 + 1]);)
    {
        int idx = 0;
        if (_data[hole * 2] > _data[hole * 2 + 1])
        {
            idx = hole * 2;
        }
        else
        {
            idx = hole * 2 + 1;
        }
        _data[hole] = _data[idx];
        _data[idx] = head;
        hole = idx;
    }
}

int Heap::size() const
{
    return _data.size();
}

void Heap::enQueue(const int &x)
{
    _data.push_back(x);
    int hole = _data.size() - 1;
    // swap the x with its parent in a recursive way but write it in a for loop manner
    for (; hole > 1 && x > _data[hole / 2]; hole /= 2)
    {
        _data[hole] = _data[hole / 2];
    }
    _data[hole] = x;
}

int Heap::deQueue()
{
    int minItem = _data[1];
    _data[1] = _data[_data.size() - 1];
    _data.pop_back();
    _percolateDown(1);
    return minItem;
}

void Heap::printQueue()
{
    std::cout << "Current Priority Queue is: ";
    for (auto i = _data.begin() + 1; i != _data.end(); ++i)
    {
        std::cout << *i << " ";
    }
    std::cout << std::endl;
}

std::vector<int> &Heap::getData()
{
    return _data;
}
