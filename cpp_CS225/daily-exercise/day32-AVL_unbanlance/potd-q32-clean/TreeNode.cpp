#include "TreeNode.h"

TreeNode* findLastUnbalanced(TreeNode* root) {
  // your code here
  return NULL;
}

void deleteTree(TreeNode* root)
{
  if (root == NULL) return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}

