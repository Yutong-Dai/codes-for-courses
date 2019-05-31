// Animal.h
#pragma once
#include <string>
using namespace std;
class Animal
{
  public:
    Animal();
    Animal(string type, string food);
    string getType();
    void setType(string newtype);
    string getFood();
    void setFood(string newfood);
    string food_;
    string print();

  private:
    string type_;
};