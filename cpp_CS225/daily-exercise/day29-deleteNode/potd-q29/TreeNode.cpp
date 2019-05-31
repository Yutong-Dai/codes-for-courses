#include "TreeNode.h"
#include <iostream>

TreeNode *&_find(TreeNode *&root, int key)
{
  if (root == NULL)
  {
    return root;
  }
  if (key == root->val_)
  {
    return root;
  }
  else
  {
    if (key < root->val_)
    {
      return _find(root->left_, key);
    }
    else
    {
      return _find(root->right_, key);
    }
  }
}

TreeNode *deleteNode(TreeNode *root, int key)
{
  // your code here
  TreeNode *&NodeToDelte = _find(root, key);
  if (NodeToDelte->left_ == NULL and NodeToDelte->right_ == NULL) // delete Leaf Node
  {
    std::cout << "LEAFT:Before deleting a node with 2 child: " << std::endl;
    inorderPrint(root);
    std::cout << std::endl;
    std::cout << "Original NodeToDelte Val: " << NodeToDelte->val_ << std::endl;
    std::cout << "LEAFT NODE CASE" << std::endl;
    delete NodeToDelte;
    NodeToDelte = NULL;
    std::cout << "After deleting a node with 2 child: " << std::endl;
    inorderPrint(root);
    std::cout << std::endl;
    return root;
  }
  else if (NodeToDelte->left_ == NULL and NodeToDelte->right_ != NULL) // delete internal Node with only one right child
  {
    std::cout << "RIGHT:Before deleting a node with 2 child: " << std::endl;
    inorderPrint(root);
    std::cout << std::endl;
    std::cout << "RIGHT: Original NodeToDelte Val: " << NodeToDelte->val_ << std::endl;
    std::cout << "RIGHT NODE CASE" << std::endl;
    TreeNode *temp = NodeToDelte->right_;
    delete NodeToDelte;
    NodeToDelte = temp;
    return root;
  }
  else if (NodeToDelte->left_ != NULL and NodeToDelte->right_ == NULL) // delete internal Node with only one left child
  {
    std::cout << "LEFT:Before deleting a node with 2 child: " << std::endl;
    inorderPrint(root);
    std::cout << std::endl;
    std::cout << "LEFT  : Original NodeToDelte Val: " << NodeToDelte->val_ << std::endl;
    std::cout << "LEFT NODE CASE" << std::endl;
    TreeNode *temp = NodeToDelte->left_;
    delete NodeToDelte;
    NodeToDelte = temp;
    return root;
  }
  else
  {
    std::cout << "TWO:Before deleting a node with 2 child: " << std::endl;
    inorderPrint(root);
    std::cout << std::endl;
    std::cout << "TWO:Original NodeToDelte Val: " << NodeToDelte->val_ << std::endl;
    TreeNode *MaxOnLeftSubTree = NodeToDelte;
    MaxOnLeftSubTree->val_ = -100;
    TreeNode *temp = NodeToDelte->left_;
    while (temp != NULL)
    {
      if (temp->val_ > MaxOnLeftSubTree->val_)
      {
        MaxOnLeftSubTree = temp;
      }
      temp = temp->right_;
    }
    // std::cout << "Now NodeToDelte Val: " << NodeToDelte->val_ << std::endl;
    std::cout << std::endl;
    std::cout << "0IOP:" << MaxOnLeftSubTree->val_ << std::endl;
    const int IOP_ORIGNAL = MaxOnLeftSubTree->val_;
    // if (MaxOnLeftSubTree->left_ == NULL)
    // {
    //   TreeNode *tempRight = NodeToDelte->right_;
    //   delete NodeToDelte;
    //   NodeToDelte = MaxOnLeftSubTree;
    //   NodeToDelte->left_ = NULL;
    //   NodeToDelte->right_ = tempRight;
    // }
    // else
    // {
    // TreeNode *MaxChild = MaxOnLeftSubTree->left_;
    // TreeNode *tempRight = NodeToDelte->right_;
    TreeNode *tempLeft = NodeToDelte->left_;
    std::cout << "1IOP:" << MaxOnLeftSubTree->val_ << std::endl;
    tempLeft = deleteNode(tempLeft, MaxOnLeftSubTree->val_);
    std::cout << "2IOP:" << MaxOnLeftSubTree->val_ << std::endl;
    NodeToDelte->val_ = MaxOnLeftSubTree->val_;
    std::cout << "3IOP:" << MaxOnLeftSubTree->val_ << std::endl;
    NodeToDelte->val_ = IOP_ORIGNAL;
    std::cout << "Update NodeToDelte Val: " << NodeToDelte->val_ << std::endl;
    NodeToDelte->left_ = tempLeft;
    // delete NodeToDelte;
    // NodeToDelte = MaxOnLeftSubTree;
    // NodeToDelte->right_ = tempRight;
    // MaxOnLeftSubTree = MaxChild;
    // NodeToDelte->left_ = tempLeft;
    // }
    std::cout << "After deleting a node with 2 child: " << std::endl;
    inorderPrint(root);
    return root;
  }
}

void inorderPrint(TreeNode *node)
{
  if (!node)
    return;
  inorderPrint(node->left_);
  std::cout << node->val_ << " ";
  inorderPrint(node->right_);
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
