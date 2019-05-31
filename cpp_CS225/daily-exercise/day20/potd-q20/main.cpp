#include <iostream>
#include <vector>

using namespace std;

#include "isValid.cpp"

int main()
{

    vector<string> tests =
        {
            // "()",
            // "()[]{}!",
            // "([cs225)]",
            // "{()}",
            // "])}",
            // "[{}](()",
            // "(",
            // "]",
            // "_",
            "([{}])",
            "_(_"
            // add your own tests here!
        };

    cout << std::boolalpha << endl;
    for (string &t : tests)
    {
        cout << t << " : ";
        cout << isValid(t);
        cout << endl
             << endl;
    }
}