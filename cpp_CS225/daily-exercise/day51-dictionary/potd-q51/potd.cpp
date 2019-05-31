#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
using namespace std;

#include <algorithm>

bool comp(const pair<string, int> &a, const pair<string, int> &b)
{
    // must have 'const', otherwise some compiler will not compile
    return a.second > b.second;
}

vector<string> topThree(string filename)
{
    string line;
    ifstream infile(filename);
    vector<string> ret;
    map<string, int> entries;
    if (infile.is_open())
    {
        while (getline(infile, line))
        {
            if (entries.find(line) == entries.end())
            {
                entries[line] = 1;
            }
            else
            {
                entries[line]++;
            }
        }
    }
    vector<pair<string, int>> entries_vec(entries.begin(), entries.end());
    sort(entries_vec.begin(), entries_vec.end(), comp);
    for (int i = 0; i < 3; i++)
    {
        ret.push_back(entries_vec[i].first);
    }
    infile.close();
    return ret;
}
