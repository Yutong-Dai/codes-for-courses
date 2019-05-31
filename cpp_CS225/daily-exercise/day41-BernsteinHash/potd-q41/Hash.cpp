#include "Hash.h"
#include <string>
#include <iostream>

unsigned long bernstein(std::string str, int M)
{
	unsigned long b_hash = 5381;
	for (const char &character : str)
	{
		b_hash *= 33;
		b_hash += character;
	}
	return b_hash % M;
}

std::string reverse(std::string str)
{
	std::string output = "";
	//Your code here
	int size = str.length() - 1;
	for (; size >= 0; size--)
	{
		output += str[size];
	}
	return output;
}
