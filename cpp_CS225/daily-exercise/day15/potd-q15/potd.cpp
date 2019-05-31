#include "potd.h"
#include <iostream>

using namespace std;

string stringList(Node *head)
{
    string outputs;
    if (head == NULL)
    {
        outputs = "Empty list";
    }
    else
    {
        Node *ptr = head;
        int count = 0;

        while (ptr->next_ != NULL)
        {
            outputs += "Node " + to_string(count) + ": " + to_string(ptr->data_) + " " + "-> ";
            ptr = ptr->next_;
            count += 1;
        }

        if (ptr->next_ == NULL)
        {
            outputs += "Node " + to_string(count) + ": " + to_string(ptr->data_);
        }
    }

    return outputs;
    // Node *ptr = head;
    // for(; ptr->next_ != NULL; ptr = ptr->next_)
}
