
#include "TreeNode.h"

#include <cstddef>
#include <iostream>
using namespace std;

TreeNode::TreeNode() : left_(NULL), right_(NULL) {}

int _getHeight(const TreeNode *subRoot)
{
  int height = 0;
  if (subRoot == NULL)
  {
    return -1;
  }
  height = std::max(_getHeight(subRoot->right_), _getHeight(subRoot->left_)) + 1;
  return height;
}

int TreeNode::getHeight()
{
  return std::max(_getHeight(this->left_), _getHeight(this->right_)) + 1;
}
