/**
 * @file avltree.cpp
 * Definitions of the binary tree functions you'll be writing for this lab.
 * You'll need to modify this file.
 */
#include <algorithm>
using std::max;
template <class K, class V>
V AVLTree<K, V>::find(const K &key) const
{
    return find(root, key);
}

template <class K, class V>
V AVLTree<K, V>::find(Node *subtree, const K &key) const
{
    if (subtree == NULL)
        return V();
    else if (key == subtree->key)
        return subtree->value;
    else
    {
        if (key < subtree->key)
            return find(subtree->left, key);
        else
            return find(subtree->right, key);
    }
}

template <class K, class V>
void AVLTree<K, V>::_updateHeight(Node *&node)
{
    node->height = max(heightOrNeg1(node->left), heightOrNeg1(node->right)) + 1;
}

template <class K, class V>
void AVLTree<K, V>::rotateLeft(Node *&t)
{
    functionCalls.push_back("rotateLeft"); // Stores the rotation name (don't remove this)
                                           // your code here
    Node *x = t->right;
    t->right = x->left;
    x->left = t;
    t = x;
    //update height
    x = t->left;
    _updateHeight(t);
    _updateHeight(x);
}

template <class K, class V>
void AVLTree<K, V>::rotateLeftRight(Node *&t)
{
    functionCalls.push_back("rotateLeftRight"); // Stores the rotation name (don't remove this)
    // Implemented for you:
    rotateLeft(t->left);
    rotateRight(t);
}

template <class K, class V>
void AVLTree<K, V>::rotateRight(Node *&t)
{
    functionCalls.push_back("rotateRight"); // Stores the rotation name (don't remove this)
                                            // your code here
    Node *x = t->left;
    t->left = x->right;
    x->right = t;
    t = x;
    // update the height
    x = t->right;
    _updateHeight(x);
    _updateHeight(t);
}

template <class K, class V>
void AVLTree<K, V>::rotateRightLeft(Node *&t)
{
    functionCalls.push_back("rotateRightLeft"); // Stores the rotation name (don't remove this)
    // your code here
    rotateRight(t->right);
    rotateLeft(t);
}

template <class K, class V>
void AVLTree<K, V>::rebalance(Node *&subtree)
{
    // your code here
    if (heightOrNeg1(subtree->right) - heightOrNeg1(subtree->left) == 2)
    { // right heavy
        if (heightOrNeg1(subtree->right->right) - heightOrNeg1(subtree->right->left) == 1)
        { // left heavy
            // Left rotation
            rotateLeft(subtree);
        }
        else
        {
            // left heavy
            // RightLeft rotation
            rotateRightLeft(subtree);
        }
    }
    else if (heightOrNeg1(subtree->right) - heightOrNeg1(subtree->left) == -2)
    { // left heavy
        if (heightOrNeg1(subtree->left->right) - heightOrNeg1(subtree->left->left) == -1)
        {
            // left heavy
            // Right rotation
            rotateRight(subtree);
        }
        else
        {
            // right heavy
            // LeftRight rotation
            rotateLeftRight(subtree);
        }
    }

    _updateHeight(subtree);
}

template <class K, class V>
void AVLTree<K, V>::insert(const K &key, const V &value)
{
    insert(root, key, value);
}

template <class K, class V>
void AVLTree<K, V>::insert(Node *&subtree, const K &key, const V &value)
{
    // your code here
    if (subtree == NULL)
    {
        // reach a leaf, add node
        subtree = new Node(key, value);
    }
    else if (subtree->key < key)
    {
        // recurse into right child
        insert(subtree->right, key, value);
    }
    else if (subtree->key > key)
    {
        // recurse into left child
        insert(subtree->left, key, value);
    }
    else
    {
        // if the key already exist
        subtree->value = value;
    }
    rebalance(subtree);
}

template <class K, class V>
typename AVLTree<K, V>::Node *&AVLTree<K, V>::_findIOP(Node *&node)
{
    if (node->right == NULL)
    {
        return node;
    }

    return _findIOP(node->right);
}

template <class K, class V>
void AVLTree<K, V>::remove(const K &key)
{
    remove(root, key);
}

template <class K, class V>
void AVLTree<K, V>::remove(Node *&subtree, const K &key)
{
    if (subtree == NULL)
        return;

    if (key < subtree->key)
    {
        // your code here
        remove(subtree->left, key);
        rebalance(subtree);
    }
    else if (key > subtree->key)
    {
        // your code here
        remove(subtree->right, key);
        rebalance(subtree);
    }
    else
    {
        if (subtree->left == NULL && subtree->right == NULL)
        {
            /* no-child remove */
            // your code here
            delete subtree;
            subtree = NULL;
        }
        else if (subtree->left != NULL && subtree->right != NULL)
        {
            /* two-child remove */
            // your code here
            Node *&temp = _findIOP(subtree->left);
            swap(temp, subtree);
            delete temp;
            temp = NULL;
            rebalance(subtree);
        }
        else
        {
            /* one-child remove */
            // your code here
            if (subtree->left != NULL)
            {
                Node *temp = subtree->left;
                delete subtree;
                subtree = temp;
            }
            else
            {
                Node *temp = subtree->right;
                delete subtree;
                subtree = temp;
            }
        }
    }
}
