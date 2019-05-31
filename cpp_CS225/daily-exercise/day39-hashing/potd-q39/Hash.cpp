#include <vector>
#include <string>
#include "Hash.h"

using namespace std;

int hashFunction(string s, int M)
{
	// Your Code Here
	//hash function to sum up the ASCII characters of the letters of the string
	int results = 0;
	for (const char &letter : s)
	{
		results += letter;
	}
	return results % M;
}

int countCollisions(int M, vector<string> inputs)
{
	int collisions = 0;
	// Your Code Here
	vector<int> hashTable(M, 0);
	for (const string &s : inputs)
	{
		int idx = hashFunction(s, M);
		hashTable[idx]++;
	}
	for (const int &times : hashTable)
	{
		if (times >= 2)
		{
			collisions += times - 1;
		}
	}
	return collisions;
}