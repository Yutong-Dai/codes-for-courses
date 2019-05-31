/**
 * @file binarytree.cpp
 * Definitions of the binary tree functions you'll be writing for this lab.
 * You'll need to modify this file.
 */
#include "TreeTraversals/InorderTraversal.h"
#include <iostream>

/**
 * @return The height of the binary tree. Recall that the height of a binary
 *  tree is just the length of the longest path from the root to a leaf, and
 *  that the height of an empty tree is -1.
 */
template <typename T>
int BinaryTree<T>::height() const
{
    // Call recursive helper function on root
    return height(root);
}

/**
 * Private helper function for the public height function.
 * @param subRoot
 * @return The height of the subtree
 */
template <typename T>
int BinaryTree<T>::height(const Node *subRoot) const
{
    // Base case
    if (subRoot == NULL)
        return -1;

    // Recursive definition
    return 1 + std::max(height(subRoot->left), height(subRoot->right));
}

/**
 * Prints out the values of the nodes of a binary tree in order.
 * That is, everything to the left of a node will be printed out before that
 * node itself, and everything to the right of a node will be printed out after
 * that node.
 */
template <typename T>
void BinaryTree<T>::printLeftToRight() const
{
    // Call recursive helper function on the root
    printLeftToRight(root);

    // Finish the line
    std::cout << std::endl;
}

/**
 * Private helper function for the public printLeftToRight function.
 * @param subRoot
 */
template <typename T>
void BinaryTree<T>::printLeftToRight(const Node *subRoot) const
{
    // Base case - null node
    if (subRoot == NULL)
        return;

    // Print left subtree
    printLeftToRight(subRoot->left);

    // Print this node
    std::cout << subRoot->elem << ' ';

    // Print right subtree
    printLeftToRight(subRoot->right);
}

/**
 * Flips the tree over a vertical axis, modifying the tree itself
 *  (not creating a flipped copy).
 */
template <typename T>
void BinaryTree<T>::mirror()
{
    //your code here
    mirror(root);
}

template <typename T>
void BinaryTree<T>::mirror(Node *&subRoot)
{
    //your code here
    if (subRoot == NULL)
    {
        return;
    }

    // Flip left subtree
    mirror(subRoot->left);

    // Flip right subtree
    mirror(subRoot->right);

    Node *temp;
    temp = subRoot->right;
    subRoot->right = subRoot->left;
    subRoot->left = temp;
}
/**
 * isOrdered() function iterative version
 * @return True if an in-order traversal of the tree would produce a
 *  nondecreasing list output values, and false otherwise. This is also the
 *  criterion for a binary tree to be a binary search tree.
 */
template <typename T>
bool BinaryTree<T>::isOrderedIterative() const
{
    // your code here
    InorderTraversal<T> iot(root);
    typename TreeTraversal<T>::Iterator it = iot.begin();
    T prev = (*it)->elem;
    for (++it; it != iot.end(); ++it)
    {
        // the comparsion is made on stack, refer to the outputs of Inorder Traversal
        if (prev > (*it)->elem)
        {
            return false;
        }
        prev = (*it)->elem;
    }

    return true;
}

/**
 * isOrdered() function recursive version
 * @return True if an in-order traversal of the tree would produce a
 *  nondecreasing list output values, and false otherwise. This is also the
 *  criterion for a binary tree to be a binary search tree.
 */
template <typename T>
bool BinaryTree<T>::isOrderedRecursive() const
{
    // your code here
    return isOrderedRecursive(root);
}

/**
 * Private helper function for the public isOrderedRecursive function
 * @param subRoot The current node in the recursion
 */
template <typename T>
bool BinaryTree<T>::isOrderedRecursive(const Node *subRoot) const
{
    bool isordered = true;

    if (subRoot->left != NULL)
    {
        isordered = isordered && (findLargest(subRoot->left) <= subRoot->elem) && isOrderedRecursive(subRoot->left);
    }
    if (subRoot->right != NULL)
    {
        isordered = isordered && (subRoot->elem <= findSmallest(subRoot->right)) && isOrderedRecursive(subRoot->right);
    }

    return isordered;
}

/**
 * Private helper function for the public isOrderedRecursive function
 * Find the largest elem of the subtree whose root is subroot
 * @param subRoot The current node in the recursion
 */
template <typename T>
T BinaryTree<T>::findLargest(const Node *subRoot) const
{
    if ((subRoot->left == NULL) && (subRoot->right == NULL))
    {
        return subRoot->elem;
    }

    T currentMax = subRoot->elem;
    if (subRoot->left != NULL)
    {
        currentMax = std::max(currentMax, findLargest(subRoot->left));
    }
    if (subRoot->right != NULL)
    {
        currentMax = std::max(currentMax, findLargest(subRoot->right));
    }

    return currentMax;
}

template <typename T>
T BinaryTree<T>::findSmallest(const Node *subRoot) const
{
    if ((subRoot->left == NULL) && (subRoot->right == NULL))
    {
        return subRoot->elem;
    }

    T currentMin = subRoot->elem;
    if (subRoot->left != NULL)
    {
        currentMin = std::min(currentMin, findSmallest(subRoot->left));
    }
    if (subRoot->right != NULL)
    {
        currentMin = std::min(currentMin, findSmallest(subRoot->right));
    }

    return currentMin;
}

/**
 * creates vectors of all the possible paths from the root of the tree to any leaf
 * node and adds it to another vector.
 * Path is, all sequences starting at the root node and continuing
 * downwards, ending at a leaf node. Paths ending in a left node should be
 * added before paths ending in a node further to the right.
 * @param paths vector of vectors that contains path of nodes
 */
template <typename T>
void BinaryTree<T>::getPaths(vector<vector<T>> &paths) const
{
    // your code here
    vector<T> path;
    paths.clear();
    getPaths(root, path, paths);
}

/**
  * Private helper functions for the public getPaths function
*/
template <typename T>
void BinaryTree<T>::getPaths(const Node *subRoot, vector<T> &path, vector<vector<T>> &paths) const
{
    // your code here
    path.push_back(subRoot->elem);

    // reaching an leaf node
    if ((subRoot->left == NULL) && (subRoot->right == NULL))
    {
        paths.push_back(path);
        path.pop_back(); // clear the path
        return;
    }

    if (subRoot->left != NULL)
    {
        getPaths(subRoot->left, path, paths);
    }

    if (subRoot->right != NULL)
    {
        getPaths(subRoot->right, path, paths);
    }

    path.pop_back(); // recursively clear the path so that we have a clean path for a next route
    return;
}

/**
 * Each node in a tree has a distance from the root node - the depth of that
 * node, or the number of edges along the path from that node to the root. This
 * function returns the sum of the distances of all nodes to the root node (the
 * sum of the depths of all the nodes). Your solution should take O(n) time,
 * where n is the number of nodes in the tree.
 * @return The sum of the distances of all nodes to the root
 */
template <typename T>
int BinaryTree<T>::sumDistances() const
{
    // your code here
    int curDepth = -1;
    return sumDistances(root, curDepth);
}

template <typename T>
int BinaryTree<T>::sumDistances(const Node *subRoot, int &curDepth) const
{
    if ((subRoot->left == NULL) && (subRoot->right == NULL))
    {
        return curDepth + 1;
    }

    int result = 0;
    curDepth++;
    if (subRoot->left != NULL)
    {
        result += sumDistances(subRoot->left, curDepth);
    }
    if (subRoot->right != NULL)
    {
        result += sumDistances(subRoot->right, curDepth);
    }

    result += curDepth;
    curDepth--;

    return result;
}