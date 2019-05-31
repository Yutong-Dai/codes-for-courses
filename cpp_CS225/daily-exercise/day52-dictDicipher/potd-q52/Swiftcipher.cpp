#include "Swiftcipher.h"
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
using namespace std;

bool comp(const pair<string, int> &a, const pair<string, int> &b)
{
	return a.second < b.second;
}

/* Swift Cipher: Frequency of words in a file corresponds to its location in the decrypted sentence */
string decipherer(string filename)
{
	string line;
	ifstream infile(filename);
	map<string, int> message;
	if (infile.is_open())
	{
		while (getline(infile, line))
		{
			if (message.find(line) == message.end())
			{
				message[line] = 1;
			}
			else
			{
				message[line]++;
			}
		}
	}
	vector<pair<string, int>> vec(message.begin(), message.end());
	sort(vec.begin(), vec.end(), comp);
	string deMessage = "";
	unsigned i = 0;
	for (; i < vec.size() - 1; i++)
	{
		deMessage += vec[i].first;
		deMessage += " ";
	}
	deMessage += vec[i].first; // no space after the last word in the message
	infile.close();
	return deMessage;
}
