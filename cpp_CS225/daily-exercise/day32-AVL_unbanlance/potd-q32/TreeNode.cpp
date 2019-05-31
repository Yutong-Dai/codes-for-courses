#include "TreeNode.h"
#include <queue>
#include <iostream>
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

void _findUnbanlanced(TreeNode *subRoot, std::queue<TreeNode *> &unbanlancedNodes)
{
  if (subRoot->left_ == NULL and subRoot->right_ == NULL)
  {
    if (!isHeightBalanced(subRoot))
    {
      unbanlancedNodes.push(subRoot);
    }
  }
  if (subRoot->left_ != NULL)
  {
    _findUnbanlanced(subRoot->left_, unbanlancedNodes);
  }
  if (subRoot->right_ != NULL)
  {
    _findUnbanlanced(subRoot->right_, unbanlancedNodes);
  }
  // std::cout << "Finish checking children, check myself..." << std::endl;
  if (!isHeightBalanced(subRoot))
  {
    // std::cout << "add myself" << subRoot->val_ << std::endl;
    unbanlancedNodes.push(subRoot);
  }
}
TreeNode *findLastUnbalanced(TreeNode *root)
{
  // your code here
  std::queue<TreeNode *> unbanlancedNodes;
  _findUnbanlanced(root, unbanlancedNodes);
  // std::cout << std::boolalpha;
  // std::cout << unbanlancedNodes.empty() << std::endl;
  // while (!unbanlancedNodes.empty())
  // {
  //   // std::cout << "hello!" << std::endl;
  //   std::cout << (unbanlancedNodes.front())->val_ << std::endl;
  //   unbanlancedNodes.pop();
  // }

  if (!unbanlancedNodes.empty())
  {
    return unbanlancedNodes.front();
  }
  else
  {
    return NULL;
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
