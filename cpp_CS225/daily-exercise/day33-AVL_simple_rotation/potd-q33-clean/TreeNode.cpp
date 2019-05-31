#include "TreeNode.h"
#include <algorithm>


void rightRotate(TreeNode* root) {

    // Your code here

}


void leftRotate(TreeNode* root) {

    // your code here

}



void deleteTree(TreeNode* root)
{
  if (root == NULL) return;
  deleteTree(root->left_);
  deleteTree(root->right_);
  delete root;
  root = NULL;
}

