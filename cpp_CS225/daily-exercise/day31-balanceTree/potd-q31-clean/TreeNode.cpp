#include "TreeNode.h"

bool isHeightBalanced(TreeNode* root) {
  // your code here
  return false;
}

void deleteTree(TreeNode* root)
{
  if (root == NULL) return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}

