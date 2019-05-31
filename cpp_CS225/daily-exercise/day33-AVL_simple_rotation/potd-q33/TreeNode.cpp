#include "TreeNode.h"
#include <algorithm>

void rightRotate(TreeNode *root)
{

  // Your code here
  // TreeNode* pivot = root;
  // TreeNode* peak = pivot->left_;
  // if (peak->right_ == NULL){
  //   peak->parent_ = pivot->parent_;
  //   peak->right_ = pivot;
  //   pivot->parent_ = peak;
  //   pivot->left_ = NULL;
  // }else{
  //   TreeNode* temp = peak->right_;
  //   peak->parent_ =
  // }
  TreeNode *pivotNode = root;
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

void leftRotate(TreeNode *root)
{

  TreeNode *pivotNode = root;
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

void deleteTree(TreeNode *root)
{
  if (root == NULL)
    return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}
