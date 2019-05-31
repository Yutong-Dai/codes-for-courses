#include "Node.h"

using namespace std;

Node *listSymmetricDifference(Node *first, Node *second) {
  // your code here
  return NULL;
}

Node::Node() {
    numNodes++;
}

Node::Node(Node &other) {
    this->data_ = other.data_;
    this->next_ = other.next_;
    numNodes++;
}

Node::~Node() {
    numNodes--;
}

int Node::numNodes = 0;
