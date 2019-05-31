#include "Node.h"
#include <iostream>
using namespace std;

Node *listSymmetricDifference(Node *first, Node *second)
{
    Node *head_to_symdiff = NULL;
    Node *current_first, *current_second, *curren_on_symdiff, *exam_duplicate;
    current_first = first;
    current_second = second;
    bool add_this_node;
    bool true_add1;
    bool true_add2;
    while (current_first != NULL)
    {
        cout << "current_first:" << current_first->data_ << endl;
        add_this_node = true;
        while (current_second != NULL)
        {
            cout << "curren_second:" << current_second->data_ << endl;

            if (current_first->data_ != current_second->data_)
            {
                current_second = current_second->next_;
            }
            else
            {
                add_this_node = false;
                break;
            }
        }
        if (add_this_node)
        {

            if (head_to_symdiff == NULL)
            {
                head_to_symdiff = new Node;
                head_to_symdiff->data_ = current_first->data_;
                head_to_symdiff->next_ = NULL;
                curren_on_symdiff = head_to_symdiff;
                cout << "Initialize head symdiff:" << head_to_symdiff->data_ << endl;
            }
            else
            {
                true_add1 = true;
                exam_duplicate = head_to_symdiff;
                while (exam_duplicate != NULL)
                {
                    if (exam_duplicate->data_ == current_first->data_)
                    {
                        true_add1 = false;
                        break;
                    }
                    exam_duplicate = exam_duplicate->next_;
                }
                if (true_add1)
                {
                    Node *addedNode = new Node;
                    addedNode->data_ = current_first->data_;
                    addedNode->next_ = NULL;
                    curren_on_symdiff->next_ = addedNode;
                    cout << "Add Node to symdiff:" << addedNode->data_ << endl;
                    curren_on_symdiff = curren_on_symdiff->next_;
                }
            }
        }
        current_second = second;
        current_first = current_first->next_;
    }
    cout << "Switch!!" << endl;
    cout << "curren_on_symdiff:" << curren_on_symdiff->data_ << endl;
    current_first = first;
    while (current_second != NULL)
    {
        cout << "current_second:" << current_second->data_ << endl;
        add_this_node = true;
        while (current_first != NULL)
        {
            cout << "curren_first:" << current_first->data_ << endl;

            if (current_second->data_ != current_first->data_)
            {
                current_first = current_first->next_;
            }
            else
            {
                add_this_node = false;
                break;
            }
        }
        if (add_this_node)
        {
            if (head_to_symdiff == NULL)
            {
                head_to_symdiff = new Node;
                head_to_symdiff->data_ = current_second->data_;
                head_to_symdiff->next_ = NULL;
                curren_on_symdiff = head_to_symdiff;
                cout << "Initialize head symdiff:" << head_to_symdiff->data_ << endl;
            }
            else
            {
                exam_duplicate = head_to_symdiff;
                true_add2 = true;
                while (exam_duplicate != NULL)
                {
                    if (exam_duplicate->data_ == current_second->data_)
                    {
                        true_add2 = false;
                        break;
                    }
                    exam_duplicate = exam_duplicate->next_;
                }
                if (true_add2)
                {
                    Node *addedNode = new Node;
                    addedNode->data_ = current_second->data_;
                    addedNode->next_ = NULL;
                    curren_on_symdiff->next_ = addedNode;
                    cout << "Add Node to symdiff:" << addedNode->data_ << endl;
                    curren_on_symdiff = curren_on_symdiff->next_;
                }
            }
        }
        current_first = first;
        current_second = current_second->next_;
    }
    return head_to_symdiff;
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

// if (curent_first->data_ == curren_second->data_)
// {
//     cout << "JUMP OUT!!" << endl;
//     break;
// }