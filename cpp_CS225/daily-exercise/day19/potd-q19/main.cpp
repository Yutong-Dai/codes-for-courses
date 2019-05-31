#include <iostream>
#include "Node.h"

void printList(Node *head)
{
  if (head == NULL)
  {
    std::cout << "Empty list" << std::endl;
    return;
  }

  Node *temp = head;
  int count = 0;
  // std::cout << temp->data_ << std::endl;
  while (temp != NULL)
  {
    std::cout << temp->data_ << std::endl;
    std::cout << "shit happens" << std::endl;
    std::cout << "Node " << std::to_string(count) << ": " << temp->data_ << std::endl;
    std::cout << temp->data_ << std::endl;
    count++;
    temp = temp->next_;
    // std::cout << "shit 2" << std::endl;
  }
}

Node *insertSorted(Node *first, int data)
{
  // your code here
  Node newNode;
  newNode.data_ = data;
  if (first == NULL)
  {
    // std::cout << newNode.data_ << std::endl;
    newNode.next_ = NULL;
    first = &newNode;
    // std::cout << "I am in~" << std::endl;
    return first;
    // std::cout << "I am bass!!" << std::endl;
  }
  else
  {
    std::cout << "I am in!!" << std::endl;
    if (first->data_ >= data)
    {
      newNode.next_ = first;
      first = &newNode;
      return first;
    }
    else
    {
      Node *current = first;
      Node *previous = NULL;
      while (current->data_ < data)
      {
        previous = current;
        if (current->next_ != NULL)
        {
          current = current->next_;
        }
        else
        {
          current->next_ = &newNode;
          newNode.next_ = NULL;
          return first;
        }
      }
      newNode.next_ = previous->next_;
      previous->next_ = &newNode;
      return first;
    }
  }
}

int main()
{
  Node n1, n3, n5;
  n1.data_ = 1;
  n3.data_ = 3;
  n5.data_ = 5;
  n1.next_ = &n3;
  n3.next_ = &n5;
  n5.next_ = NULL;
  Node *first = NULL; //&n1;
  std::cout << "First List:" << std::endl;
  printList(first);
  // std::cout << ((first == NULL) ? "yes" : "no") << std::endl;
  std::cout << "insertSort:" << std::endl;
  Node *newlist = insertSorted(first, 10);
  printList(newlist);

  return 0;
}
