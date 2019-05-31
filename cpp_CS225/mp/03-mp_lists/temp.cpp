#include "List.h"
#include <iostream>

int main()
{
    List<unsigned> list;
    list.insertFront(1);
    list.insertFront(2);
    list.insertFront(3);
    std::cout << "yes" << std::endl;
    std::cout << *list.begin() << std::endl;
    std::cout << list.end() << std::endl;
    return 0;
}
