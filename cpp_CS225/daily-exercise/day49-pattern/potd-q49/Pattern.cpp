#include "Pattern.h"
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

using namespace std;

bool wordPattern(std::string pattern, std::string str)
{
    //write your code here
    map<char, string> patternMap;
    vector<string> words;
    // make it easier to get the last word
    // each word in the string in the form of "word ".
    str = str + " ";
    // int count = 0;
    for (char &key : pattern)
    {
        // cout << count << endl;
        size_t endPosition = str.find(" ");
        // cout << "end position: " << endPosition << endl;
        size_t size = str.size();
        string word;
        word = str.substr(0, endPosition);
        // cout << "Checking key:" << key << " word:" << word << endl;
        // update the string (pass by object no worry)
        str = str.substr(endPosition + 1, size);

        if (patternMap.find(key) == patternMap.end())
        {
            // haven't build this key into the dictionary
            if (find(words.begin(), words.end(), word) == words.end())
            {
                // if the word has never appeared
                patternMap[key] = word;
                words.push_back(word);
            }
            else
            {
                // new key, old word, return false
                cout << "new key, old word, return false" << endl;
                return false;
            }
        }
        else
        {
            if (patternMap[key] != word)
            {
                // old key, new word, return false
                cout << "old key, new word, return false" << endl;
                return false;
            }
        }
    }
    cout << "will return true" << endl;
    return true;
}
