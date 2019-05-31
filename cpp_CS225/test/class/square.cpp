#include "square.h"
#include <iostream>
Square::Square()
{
    width_ = 2;
}

Square::Square(double w)
{
    width_ = w;
}

double Square::getArea()
{
    return width_ * width_;
}

void Square::print_length(int length)
{
    std::cout << length << std::endl;
}