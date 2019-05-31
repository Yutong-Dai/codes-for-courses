#include "TreeNode.h"

TreeNode::RotationType balanceTree(TreeNode *&subroot)
{
  // Your code here
  if (leftHeavy(subroot))
  {
    TreeNode *keyPoint = subroot->left_;
    if (leftHeavy(keyPoint))
    {
      return TreeNode::right;
    }
    else
    {
      return TreeNode::leftRight;
    }
  }
  else
  {
    TreeNode *keyPoint = subroot->right_;
    if (rightHeavy(keyPoint))
    {
      return TreeNode::left;
    }
    else
    {
      return TreeNode::rightLeft;
    }
  }
}
