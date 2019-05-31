// Your code here
#include "potd.h"
string getFortune(const string &s)
{
    string fortunes[5];
    fortunes[0] = "As you wish!";
    fortunes[1] = "Nec spe nec metu!";
    fortunes[2] = "Do, or do not. There is no try.";
    fortunes[3] = "This fortune intentionally left blank.";
    fortunes[4] = "Amor Fati!";

    return fortunes[s.length() % 5];
}