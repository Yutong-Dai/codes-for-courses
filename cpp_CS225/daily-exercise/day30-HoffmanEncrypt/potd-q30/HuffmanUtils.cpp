#include "HuffmanNode.h"
#include "HuffmanUtils.h"
#include <string>
#include <vector>
#include <map>
using std::map;
using std::string;
using std::vector;

/**
 * binaryToString
 *
 * Write a function that takes in the root to a huffman tree
 * and a binary string.
 *
 * Remember 0s in the string mean left and 1s mean right.
 */

string binaryToString(string binaryString, HuffmanNode *huffmanTree)
{
    /* TODO: Your code here */
    HuffmanNode *current = huffmanTree;
    string decodedString;
    for (const char &c : binaryString)
    {
        if (c == '0')
        {
            current = current->left_;
        }
        else
        {
            current = current->right_;
        }
        if (current->left_ == NULL and current->right_ == NULL)
        {
            decodedString += current->char_;
            current = huffmanTree;
        }
    }
    return decodedString;
}

/**
 * stringToBinary
 *
 * Write a function that takes in the root to a huffman tree
 * and a character string. Return the binary representation of the string
 * using the huffman tree.
 *
 * Remember 0s in the binary string mean left and 1s mean right
 */

void _findpath(vector<vector<string>> &path_built, vector<string> &path_to_build, vector<char> &keys, HuffmanNode *subRoot)
{
    if (subRoot->left_ == NULL and subRoot->right_ == NULL)
    {
        path_built.push_back(path_to_build);
        keys.push_back(subRoot->char_);
        path_to_build.pop_back();
        return;
    }
    if (subRoot->left_ != NULL)
    {
        path_to_build.push_back("0");
        _findpath(path_built, path_to_build, keys, subRoot->left_);
    }
    if (subRoot->right_ != NULL)
    {
        path_to_build.push_back("1");
        _findpath(path_built, path_to_build, keys, subRoot->right_);
    }
    if (!path_to_build.empty())
    {
        path_to_build.pop_back();
    }
}

string stringToBinary(string charString, HuffmanNode *huffmanTree)
{
    /* TODO: Your code here */
    string encodedString;
    vector<vector<string>> path_built;
    vector<string> path_to_build;
    vector<char> keys;
    _findpath(path_built, path_to_build, keys, huffmanTree);

    map<char, string> huffmanTreeDict;
    for (unsigned i = 0; i < keys.size(); i++)
    {
        path_to_build = path_built.at(i);
        string path_to_build_devectorize = "";
        for (const string &c : path_to_build)
        {
            path_to_build_devectorize += c;
        }
        huffmanTreeDict[keys.at(i)] = path_to_build_devectorize;
    }

    for (const char &c : charString)
    {
        encodedString += huffmanTreeDict.at(c);
    }

    return encodedString;
}
