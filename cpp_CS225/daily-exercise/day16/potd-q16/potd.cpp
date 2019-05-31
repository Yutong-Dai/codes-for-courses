#include "potd.h"
#include <iostream>
using namespace std;

void insertSorted(Node **head, Node *item)
{
  if (*head == NULL)
  {
    *head = item;
    cout << "intialize head" << endl;
  }
  else
  {
    if (item->data_ <= (*head)->data_)
    {
      item->next_ = (*head);
      *head = item;
      cout << "Insert at the front" << endl;
    }
    else
    {
      Node *current = *head;
      Node *previous = NULL;
      bool isENd = false;
      while (item->data_ > current->data_)
      {
        if (current->next_ == NULL)
        {
          current->next_ = item;
          item->next_ = NULL;
          cout << "Insert at the back" << endl;
          isENd = true;
          break;
        }
        else
        {
          previous = current;
          current = current->next_;
        }
      }
      if (!isENd)
      {
        item->next_ = previous->next_;
        previous->next_ = item;
        cout << "Insert at the middle" << endl;
      }
    }
  }
}