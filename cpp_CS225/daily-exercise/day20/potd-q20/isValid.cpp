#include <string>
#include <stack>

using namespace std;

bool isValid(string input)
{
    // stack<char> st;
    stack<char> brackets;
    char temp;
    char shit;
    string input_ = "";
    for (unsigned int i = 0; i < input.length(); i++)
    {
        shit = input[i];
        if (shit == '(' or shit == ')' or shit == '[' or shit == ']' or shit == '{' or shit == '}')
        {
            input_ += shit;
        }
    }
    input = input_;
    for (unsigned int i = 0; i < input.length(); i++)
    {
        char stack_out = input[i];
        if (stack_out == '(' or stack_out == '[' or stack_out == '{')
        {
            brackets.push(stack_out);
            continue;
        }
        if (brackets.empty())
        {
            if (stack_out == ')' or stack_out == ']' or stack_out == '}')
            {
                return false;
            }
            return true;
        }
        switch (stack_out)
        {
        case (')'):
            temp = brackets.top();
            brackets.pop();
            if (temp == '{' or temp == '[')
            {
                return false;
            }
            break;

        case ('}'):
            temp = brackets.top();
            brackets.pop();
            if (temp == '(' or temp == '[')
            {
                return false;
            }
            break;

        case (']'):
            temp = brackets.top();
            brackets.pop();
            if (temp == '(' or temp == '{')
            {
                return false;
            }
            break;
        }
    }
    return (brackets.empty()) ? true : false;
}
