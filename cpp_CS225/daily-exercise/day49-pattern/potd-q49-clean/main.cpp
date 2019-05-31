#include "Pattern.h"
#include <string>
#include <iostream>
#include <sstream>

int main(){

    std::string s1= "dog cat cat cat cat cat cat cat dog";
    std::string p1 = "abbbbbbba";
    std::string s2 = "fish dog dog dog dog dog dog cat";
    std::string p2 = "abbbbbba";

    std::cout << " The pattern p1 matches string s1 " << wordPattern(p1,s1)<< std::endl;
    std::cout << " The pattern p2 matches string s2 " << wordPattern(p2,s2)<< std::endl;
 
    return 0;

}


