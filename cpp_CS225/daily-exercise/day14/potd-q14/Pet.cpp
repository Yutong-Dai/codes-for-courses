// Pet.cpp
#include "Pet.h"

Pet::Pet()
{
    setType("cat");
    food_ = "fish";
    name_ = "Fluffy";
    owner_name_ = "Cinda";
}

Pet::Pet(string type, string food, string name, string owner_name) : name_(name), owner_name_(owner_name)
{
    setType(type);
    food_ = food;
}
void Pet::setFood(string food)
{
    food_ = food;
}

string Pet::getFood()
{
    return food_;
}
void Pet::setName(string name)
{
    name_ = name;
}
string Pet::getName()
{
    return name_;
}
void Pet::setOwnerName(string owner_name)
{
    owner_name_ = owner_name;
}
string Pet::getOwnerName()
{
    return owner_name_;
}
string Pet::print()
{
    return "My name is " + name_;
}