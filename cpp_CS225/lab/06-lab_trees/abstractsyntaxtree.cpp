#include "abstractsyntaxtree.h"

/**
 * Calculates the value from an AST (Abstract Syntax Tree). To parse numbers from strings, please use std::stod
 * @return A double representing the calculated value from the expression transformed into an AST
 */
double AbstractSyntaxTree::eval() const
{
    // @TODO Your code goes here...
    return eval(getRoot());
}

/**
 * Private helper function for the public eval function
 * @param node The current node
**/
double AbstractSyntaxTree::eval(typename BinaryTree<std::string>::Node *node) const
{
    if ((node->left == NULL) && (node->right == NULL))
    {
        // leaf node must be a double number, cannot be an operator
        return stod(node->elem);
    }

    double lhs, rhs;

    if (node->left != NULL)
    {
        lhs = eval(node->left);
    }

    if (node->right != NULL)
    {
        rhs = eval(node->right);
    }

    if (node->elem == "+")
    {
        return lhs + rhs;
    }
    else if (node->elem == "-")
    {
        return lhs - rhs;
    }
    else if (node->elem == "*")
    {
        return lhs * rhs;
    }
    else
    {
        // (node->elem == "/")
        return lhs / rhs;
    }
}
