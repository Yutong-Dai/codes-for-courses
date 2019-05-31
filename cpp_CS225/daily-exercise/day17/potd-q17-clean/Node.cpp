#include "Node.h"

void sortList(Node **head) {
  // your code here!

  /*
  Selection sort:
  Repeatedly moving the minimum node
  to the beginning
  */


  // Some variables you probably will need
  Node *minNode = NULL;
  Node *minPrev = NULL;

  if (*head == NULL)
      return;

  int minData = (*head)->data_;
  Node *tmp = *head;
  Node *prev = NULL;

  while ( tmp != NULL ) {
      // TODO: Find the minimum node
  }

  if (minPrev != NULL && minNode != NULL) {
      // TODO: Move the minimum node to the beginning
      // and update head
  }

  // Recurse on the rest of the list
  sortList(& (*head)->next_);
}

Node::Node() {
    numNodes++;
}

Node::Node(const Node &other) {
    this->data_ = other.data_;
    this->next_ = other.next_;
    numNodes++;
}

Node::~Node() {
    numNodes--;
}

int Node::numNodes = 0;
