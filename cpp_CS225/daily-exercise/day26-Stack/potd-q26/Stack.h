#pragma once

#include <cstddef>
#include <vector>
class Stack
{
public:
  int size() const;
  bool isEmpty() const;
  void push(int value);
  int pop();

private:
  std::vector<int> mystatck;
};