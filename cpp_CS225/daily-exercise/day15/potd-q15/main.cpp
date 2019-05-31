#include "potd.h"
#include <iostream>
using namespace std;

int main()
{
  // Test 1: An empty list
  Node *head = NULL;
  cout << stringList(head) << endl;

  // Test 2: A list with exactly one node
  Node node1;
  node1.data_ = 100;
  node1.next_ = NULL;
  head = &node1;
  cout << stringList(head) << endl;
  // Test 3: A list with more than one node
  Node node2, node3;
  node1.next_ = &node2;
  node2.data_ = 200;
  node2.next_ = &node3;
  node3.data_ = 300;
  node3.next_ = NULL;
  cout << stringList(head) << endl;
  return 0;
}
