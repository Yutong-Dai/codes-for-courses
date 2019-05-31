#include "Node.h"

/** Create a new node and add it to the list, while keeping the list sorted.
*/

/** Creates a new list (containing new nodes, allocated on the heap)
	that contains the set union of the values in both lists.
*/
Node *listUnion(Node *first, Node *second)
{
    Node *out = NULL;

    while (first != NULL)
    {
        // your code here
        // hint: call insertSorted and update l1
    }

    while (second != NULL)
    {
        // your code here
    }

    return out;
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
