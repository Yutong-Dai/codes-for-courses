#include "Memory.h"
#include <iostream>
#include <iomanip>
#include "Node.h"
#include <vector>
#include <utility>

using namespace std;

/**
Frees the given memory address. Returns if the free was successful or not
Be sure to merge adjacent free blocks when possible
*/

/**
 *  If our memory is 16 bytes in total and every Node can hold 4 bytes. So we have 4 Nodes, i.e.,  {1,2,3,4} in total.
 * A chunk can be regared as a subset of the total nodes, where each elements are in succissive order, i.e. {1,2} {2,3,4}.
 * So mermging free adjacent memory is essentially forming new chunks. For example, 
 * Node1, Node4: inUse -> we have two small chunks of size 4 byets each
 * Node2, Node3: notinUse -> we have one big chunk of size 8
 * Implementation:
 *       Node1->Next = Node2; 
 *       Node2->Next = Node4; Node4->Prev = Node2
 *       delete Node3
 * So the memory addresses change from (00, 04, 08, 12) to (00, 04, 12), where the middle one refers to a large chunk.
*/

bool Memory::free(unsigned address)
{
    Node *current = head;
    while (current != NULL)
    {
        if (current->address == address)
        {
            // Your code there
            if ((current->previous == NULL) && (current->next == NULL))
            {
                current->inUse = false;
            }
            else if (current->previous == NULL)
            {   // if current is head
                // next must not be NULL
                if ((current->next)->inUse)
                { // if next is inUse
                    current->inUse = false;
                }
                else
                { // if next is not inUse
                    Node *temp = current->next;
                    current->inUse = false;
                    current->next = temp->next;
                    (temp->next)->previous = current;
                    delete temp;
                }
            }
            else if (current->next == NULL)
            { // if current is tail
                if ((current->previous)->inUse)
                { // if previous is inUse
                    current->inUse = false;
                }
                else
                { // if previous is not inUse
                    (current->previous)->next = NULL;
                    delete current;
                }
            }
            else
            {
                if ((current->next)->inUse && (current->previous)->inUse)
                {
                    current->inUse = false;
                }
                else if ((current->next)->inUse)
                { // next is inUse means previous in NOT inUse
                    (current->previous)->next = current->next;
                    (current->next)->previous = current->previous;
                    delete current;
                }
                else if ((current->previous)->inUse)
                { // next is not inUse; if previous is inUse, then do following
                    current->inUse = false;
                    Node *temp = current->next;
                    current->next = temp->next;
                    (temp->next)->previous = current;
                    delete temp;
                }
                else
                { // both next and previous are not in use
                    Node *temp = current->next;
                    (current->previous)->next = temp->next;
                    (temp->next)->previous = current->previous;
                    delete temp;
                    delete current;
                }
            }

            return true;
        }
        current = current->next;
    }

    return false;
}

/**
Reorganizes memory structure so that all of the allocated memory is grouped at the bottom (0x0000) and there is one large
chunk of free memory above.

Note: We do not care about the order of the allocated memory chunks
*/
void Memory::defragment()
{
    Node *current = head;
    while (current != NULL)
    {

        if (current->inUse)
        {
            // Do nothing
        }
        else
        {
            // TODO: Find the next chunk of allocated memory and shift it down to current's address
            // We recommend using the helper function `findNextAllocatedMemoryChunk` and `getSize`
            // Your code here
            Node *nextInUse = findNextAllocatedMemoryChunk(current);
            if (nextInUse != NULL)
            {
                Node *temp1, *temp2;
                temp1 = current->next;
                current->next = nextInUse;
                while (temp1 != nextInUse)
                {
                    temp2 = temp1->next;
                    delete temp1;
                    temp1 = temp2;
                }
                // Now current and nextInUse are adjacent
                size_t thisSize = getSize(nextInUse);
                nextInUse->address = current->address + thisSize;
                current->inUse = true;
                nextInUse->inUse = false;
            }
        }

        current = current->next;
    }

    // TODO: Finally merge all of the free blocks of memory together
    Node *FirstFree;
    Node *nextInUse = findNextAllocatedMemoryChunk(head);
    if (nextInUse == NULL)
    {
        FirstFree = head;
    }
    else
    {
        Node *temp;
        temp = head;
        while (temp != NULL)
        {
            FirstFree = temp;
            temp = findNextAllocatedMemoryChunk(temp);
        }
        FirstFree = FirstFree->next;
    }
    if (FirstFree->next != NULL)
    {
        Node *temp3 = FirstFree->next;
        Node *temp4;
        FirstFree->next = NULL;
        while (temp3 != NULL)
        {
            temp4 = temp3->next;
            delete temp3;
            temp3 = temp4;
        }
    }
}
