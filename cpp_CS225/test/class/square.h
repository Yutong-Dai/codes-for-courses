#pragma once
class Square
{
private:
  /* data */
  double width_;

public:
  Square();
  Square(double w);
  double getArea();
  void print_length(int);
};
