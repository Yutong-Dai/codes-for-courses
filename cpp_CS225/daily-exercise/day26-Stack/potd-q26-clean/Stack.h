#pragma once

#include <cstddef>

class Stack {
public:
  int size() const;
  bool isEmpty() const;
  void push(int value);
  int pop();
};