// Your code here
#ifndef _FOOD_H
#define _FOOD_H

#include <string>
class Food
{
  public:
    Food();
    std::string get_name();
    void set_name(std::string new_name);
    unsigned int get_quantity();
    void set_quantity(unsigned int new_quantity);

  private:
    std::string name_;
    unsigned int quantity_;
};

#endif