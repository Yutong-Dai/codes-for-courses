// your code here
#include <iostream>
#include "Food.h"
#include "q5.h"

int main()
{
    Food f;
    std::cout << "You have " << f.get_quantity() << " " << f.get_name() << "." << std::endl;
    Food *ptr = &f;
    increase_quantity(ptr);
    std::cout << "You have " << f.get_quantity() << " " << f.get_name() << "." << std::endl;
    return 0;
}
