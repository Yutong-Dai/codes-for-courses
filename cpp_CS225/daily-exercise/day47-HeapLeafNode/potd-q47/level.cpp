#include "MinHeap.h"
#include <math.h>
#include <iostream>
using namespace std;

vector<int> lastLevel(MinHeap &heap)
{
        // Your code here
        unsigned int size = heap.elements.size() - 1;
        unsigned int beginningIdx = pow(2, floor(log2(size)));
        cout << "size: " << size << endl;
        cout << "beginningIdx: " << beginningIdx << endl;
        vector<int> leafNode;
        for (unsigned int i = beginningIdx; i < heap.elements.size(); i++)
        {
                cout << "Push back: " << heap.elements[i] << endl;
                leafNode.push_back(heap.elements[i]);
        }
        return leafNode;
}
