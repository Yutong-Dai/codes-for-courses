#include "TreeNode.h"
// #include <algorithm>
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
  if (!isHeightBalanced(subRoot))
  {
    unbanlancedNodes.push(subRoot);
  }
}
TreeNode *findLastUnbalanced(TreeNode *root)
{
  std::queue<TreeNode *> unbanlancedNodes;
  _findUnbanlanced(root, unbanlancedNodes);
  if (!unbanlancedNodes.empty())
  {
    return unbanlancedNodes.front();
  }
  else
  {
    return NULL;
  }
}

void rightRotate(TreeNode *root)
{

  // Your code here
  TreeNode *pivotNode = findLastUnbalanced(root);
  if (pivotNode != NULL)
  {
    TreeNode *pivotParent = pivotNode->parent_;
    TreeNode *peak = pivotNode->left_;
    if (pivotParent != NULL)
    {
      peak->parent_ = pivotParent;
      pivotParent->left_ = peak;
    }
    else
    {
      peak->parent_ = NULL;
      root = peak;
    }
    if (peak->right_ == NULL)
    {
      peak->right_ = pivotNode;
      pivotNode->left_ = NULL;
    }
    else
    {
      TreeNode *rightChild = peak->right_;
      peak->right_ = pivotNode;
      pivotNode->left_ = rightChild;
      rightChild->parent_ = pivotNode;
    }
    pivotNode->parent_ = peak;
  }
}

void leftRotate(TreeNode *root)
{

  // your code here
  TreeNode *pivotNode = findLastUnbalanced(root);
  if (pivotNode != NULL)
  {
    TreeNode *pivotParent = pivotNode->parent_;
    TreeNode *peak = pivotNode->right_;
    if (pivotParent != NULL)
    {
      peak->parent_ = pivotParent;
      pivotParent->right_ = peak;
    }
    else
    {
      peak->parent_ = NULL;
      root = peak;
    }
    if (peak->left_ == NULL)
    {
      peak->left_ = pivotNode;
      pivotNode->right_ = NULL;
    }
    else
    {
      TreeNode *leftChild = peak->left_;
      peak->left_ = pivotNode;
      pivotNode->right_ = leftChild;
      leftChild->parent_ = pivotNode;
    }
    pivotNode->parent_ = peak;
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
