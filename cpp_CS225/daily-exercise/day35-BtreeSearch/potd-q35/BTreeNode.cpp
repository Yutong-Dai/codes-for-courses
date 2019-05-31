#include <vector>
#include "BTreeNode.h"

BTreeNode *find(BTreeNode *root, int key)
{
  // Your Code Here
  unsigned int i;
  for (i = 0; i < root->elements_.size() and root->elements_[i] < key; i++)
  {
  }

  if (i < (root->elements_).size() and (root->elements_[i]) == key)
  {
    return root;
  }

  if (root->is_leaf_)
  {
    return NULL;
  }

  return find((root->children_)[i], key);
}
