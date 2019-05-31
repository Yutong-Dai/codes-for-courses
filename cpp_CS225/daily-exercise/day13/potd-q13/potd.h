//
// Write a templated function `more` which takes in two variables of the same
// type and returns whichever variable is greater than (`>`) the other.
//
#pragma once
// Because the function argument passed in for type T could be a class type, and it’s generally not a good idea to pass classes by value, it would be better to make the parameters and return types of our templated function const references
template <typename T>
const T &more(const T &one, const T &two)
{
    return (one > two) ? one : two;
}
