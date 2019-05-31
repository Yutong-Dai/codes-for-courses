#include "TreeNode.h"
#include <algorithm>

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

int _getBanlance(const TreeNode *subRoot)
{
  int heightRight = _getHeight(subRoot->right_) + 1;
  int heightLeft = _getHeight(subRoot->left_) + 1;
  return (heightRight - heightLeft);
}

bool isHeightBalanced(TreeNode *root)
{
  int b = _getBanlance(root);
  if (b < -1 or b > 1)
  {
    return false;
  }
  else
  {
    return true;
  }
}

void deleteTree(TreeNode *root)
{
  if (root == NULL)
    return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}
