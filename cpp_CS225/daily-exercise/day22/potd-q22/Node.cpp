#include "Node.h"
#include <iostream>
using namespace std;

Node *listIntersection(Node *first, Node *second)
{
    Node *first_current = first;
    Node *second_current = second;
    Node *intersection_current = NULL;
    Node *head = NULL;
    Node *temp = NULL;
    while (first_current != NULL)
    {
        std::cout << "first current data" << first_current->data_ << std::endl;
        while (second_current != NULL)
        {
            std::cout << "second current data" << second_current->data_ << std::endl;
            if (head == NULL)
            {
                if (first_current->data_ == second_current->data_)
                {
                    // std::cout << "create head" << std::endl;
                    head = new Node;
                    head->data_ = second_current->data_;
                    head->next_ = NULL;
                    intersection_current = head;
                    break;
                }
            }
            else
            {
                if (first_current->data_ == second_current->data_)
                {
                    // std::cout << "create sth" << std::endl;
                    temp = new Node;
                    temp->data_ = second_current->data_;
                    temp->next_ = NULL;
                    intersection_current->next_ = temp;
                    intersection_current = intersection_current->next_;
                    break;
                }
            }
            second_current = second_current->next_;
        }

        first_current = first_current->next_;
        second_current = second;
        // std::cout << "reset second_current" << std::endl;
    }

    return head;
}

Node::Node()
{
    numNodes++;
}

Node::Node(Node &other)
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
