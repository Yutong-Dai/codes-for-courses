#include "potd.h"
#include <vector>
#include <iostream>
using namespace std;

int main()
{
	vector<string> mytest = topThree("in1.txt");
	for (unsigned i = 0; i < mytest.size(); i++)
	{
		std::cout << mytest[i] << std::endl;
	}
	return 0;
}
