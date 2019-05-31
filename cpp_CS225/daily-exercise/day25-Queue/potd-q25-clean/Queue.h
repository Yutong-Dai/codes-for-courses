#pragma once

#include <cstddef>

class Queue {
    public:
        int size() const;
        bool isEmpty() const;
        void enqueue(int value);
        int dequeue();
};