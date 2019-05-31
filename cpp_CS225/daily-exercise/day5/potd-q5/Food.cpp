// Your code here
#include "Food.h"

Food::Food()
{
    name_ = "Napa";
    quantity_ = 1;
}
std::string Food::get_name()
{
    return name_;
}
void Food::set_name(std::string new_name)
{
    name_ = new_name;
}
unsigned int Food::get_quantity()
{
    return quantity_;
}
void Food::set_quantity(unsigned int new_quantity)
{
    quantity_ = new_quantity;
}