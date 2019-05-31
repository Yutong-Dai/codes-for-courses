#include <vector>
#include "BTreeNode.h"
#include <iostream>
using namespace std;

void traverse_(BTreeNode *subRoot, std::vector<int> &v)
{
    BTreeNode *current = subRoot;
    if (current->is_leaf_)
    {
        // cout << "size: " << (current->elements_).size() << endl;
        cout << "Adding:" << endl;
        for (unsigned i = 0; i < (current->elements_).size(); i++)
        {
            //cout << "i value: " << i << endl;
            cout << current->elements_[i] << endl;
            v.push_back(current->elements_[i]);
        }
    }
    else
    {
        for (unsigned j = 0; j < (current->children_).size(); j++)
        {
            BTreeNode *currentChild = (current->children_)[j];
            cout << "checking currentChild with element:";
            for (unsigned k = 0; k < currentChild->elements_.size(); k++)
            {
                cout << currentChild->elements_[k] << endl;
            }
            traverse_(currentChild, v);
            cout << "Finishing the currentChild" << endl;
        }

        for (unsigned t = 0; t < (current->elements_).size(); t++)
        {
            v.push_back(current->elements_[t]);
        }
        // cout << "Current:" << current->elements_[0] << endl;
        // cout << "Current:" << current->elements_[1] << endl;
        sort(v.begin(), v.end());
        // cout << "Finishing the children" << endl;
    }
}
std::vector<int> traverse(BTreeNode *root)
{
    // your code here
    std::vector<int> v;
    traverse_(root, v);
    return v;
}
