#include <iostream>
#include <vector>
#include "Heap.h"
#include <iostream>
using namespace std;

int main()
{
        std::vector<int> data = {-65536, 12, 7, 8, 13, 4, -1, 6, 5, 10, 3, 1, 15};

        Heap h;
        std::vector<int>::iterator it = data.begin();
        for (; it != data.end(); it++)
        {
                h.enQueue(*it);
        }

        h.printQueue();

        while (h.size() != 1)
        {
                std::cout << h.deQueue() << " Pop from Priority Queue" << std::endl;
                // h.printQueue();
        }
        return 0;
}
