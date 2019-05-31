#include <vector>
#include <string>

vector<string> NoProblem(int start, vector<int> created, vector<int> needed)
{

    // your code here
    unsigned size = created.size();
    vector<string> results(size);
    int totalProblems = start;
    for (unsigned i = 0; i < size; i++)
    {
        if (totalProblems >= needed[i])
        {
            results[i] = "No problem! :D";
            totalProblems += created[i];
            totalProblems -= needed[i];
        }
        else
        {
            results[i] = "No problem. :(";
            totalProblems += created[i];
        }
    }
    return results;
}
