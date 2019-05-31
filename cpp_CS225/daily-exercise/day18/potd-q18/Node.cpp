#include "Node.h"
#include <iostream>
using namespace std;

void mergeList(Node *first, Node *second)
{
    // your code here!
    if (first != NULL and second != NULL)
    {
        Node *current1 = first;
        Node *current2 = second;
        Node *temp1, *temp2;
        bool isContinue = true;
        while (isContinue)
        {
            // cout << (*current1->next_).data_ << endl;
            if (current1->next_ == NULL or current2->next_ == NULL)
            {
                isContinue = false;
            }
            if (current1->next_ == NULL and current2->next_ != NULL)
            {
                current1->next_ = current2;
            }
            else if (current1->next_ != NULL and current2->next_ == NULL)
            {
                current2->next_ = current1->next_;
                current1->next_ = current2;
            }
            else if (current1->next_ == NULL and current2->next_ == NULL)
            {
                current1->next_ = current2;
            }
            else
            {
                cout << "5" << endl;
                temp1 = current1->next_;
                temp2 = current2->next_;
                current2->next_ = current1->next_;
                current1->next_ = current2;
                current1 = temp1;
                current2 = temp2;
            }
        }
    }
}

Node::Node()
{
    numNodes++;
}

Node::Node(const Node &other)
{
    this->data_ = other.data_;
    this->next_ = other.next_;
    numNodes++;
}

Node::~Node()
{
    numNodes--;
}

int Node::numNodes = 0;
