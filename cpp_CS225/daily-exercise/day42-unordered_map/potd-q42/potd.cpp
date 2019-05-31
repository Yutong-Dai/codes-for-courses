#include <unordered_map>

using namespace std;

unordered_map<string, int> common_elems(unordered_map<string, int> &mapA,
                                        unordered_map<string, int> &mapB)
{

    // your code here
    unordered_map<string, int> common;
    auto iterA = mapA.begin();
    while (iterA != mapA.end())
    {
        // find returns an iterator to the element, if the specified key value is found,
        // or unordered_map::end if the specified key is not found in the container.
        auto iterB = mapB.find(iterA->first);
        if (iterB != mapB.end())
        {
            common[iterA->first] = iterA->second + iterB->second; // iterA->first: get key; iterA->second: get value
            iterA = mapA.erase(iterA);
            mapB.erase(iterB);
        }
        else
        {
            iterA++;
        }
    }
    return common;
}