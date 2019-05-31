#include <vector>
#include "BTreeNode.h"
#include <iostream>
using namespace std;

std::vector<int> traverse(BTreeNode *root)
{
    // your code here
    std::vector<int> v;
    if (root->is_leaf_)
    {
        // base case
        v.assign((root->elements_).begin(), (root->elements_).end());
        cout << "Adding:";
        for (auto x : v)
        {
            cout << x << " ";
        }
        cout << endl;
        return v;
    }

    unsigned i = 0;

    std::vector<int> temp = traverse((root->children_)[i]);

    cout << "Current temp ";
    for (auto x : temp)
    {
        cout << x << " ";
    }
    cout << endl;
    cout << "Current v:";
    for (auto x : v)
    {
        cout << x << " ";
    }
    cout << endl;
    v.assign(temp.begin(), temp.end());

    for (; i < (root->elements_).size(); i++)
    {
        cout << "Current adding:" << (root->elements_)[i] << endl;
        v.push_back((root->elements_)[i]);
        std::vector<int> temp = traverse((root->children_)[i + 1]);
        v.insert(v.end(), temp.begin(), temp.end());
        cout << "Current inserting:";
        for (auto x : temp)
        {
            cout << x << " ";
        }
        cout << endl;
    }

    return v;
}
