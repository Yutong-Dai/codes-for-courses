// Animal.cpp
#include "Animal.h"

Animal::Animal()
{
    type_ = "cat";
    food_ = "fish";
}

Animal::Animal(string type, string food)
{
    type_ = type;
    food_ = food;
}

string Animal::getFood()
{
    return food_;
}

void Animal::setFood(string newfood)
{
    food_ = newfood;
}

string Animal::getType()
{
    return type_;
}

void Animal::setType(string newtype)
{
    type_ = newtype;
}

string Animal::print()
{
    return "I am a " + this->getType() + ".";
}