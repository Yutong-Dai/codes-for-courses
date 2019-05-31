#include "Hash.h"
#include <string>

unsigned long bernstein(std::string str, int M)
{
	unsigned long b_hash = 5381;
	//Your code here
	return b_hash % M;
}

std::string reverse(std::string str)
{
    std::string output = "";
	//Your code here

	return output;
}
