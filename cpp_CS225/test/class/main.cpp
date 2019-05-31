#include "square.h"
#include <iostream>
#include <stdlib.h>

int main()
{
    Square s1;
    Square s2(10);
    std::cout << "S1 area:" << s1.getArea() << "|| S2 area:" << s2.getArea() << std::endl;
    s2.print_length(5555);
    return 1;
}