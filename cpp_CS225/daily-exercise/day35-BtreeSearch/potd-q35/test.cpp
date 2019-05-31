#include <vector>
#include <iostream>

int main()
{
    std::vector<int> myvec;
    myvec.push_back(1);
    myvec.push_back(3);
    myvec.push_back(5);
    unsigned i;

    std::cout << "myvec size:" << myvec.size() << std::endl;

    int key = -1;
    for (i = 0; i < myvec.size() && myvec[i] < key; i++)
    {
    }
    std::cout << "key:" << key << ", i value:" << i << std::endl;
    key = 2;
    for (i = 0; i < myvec.size() && myvec[i] < key; i++)
    {
    }
    std::cout << "key:" << key << ", i value:" << i << std::endl;

    key = 3;
    for (i = 0; i < myvec.size() && myvec[i] < key; i++)
    {
    }
    std::cout << "key:" << key << ", i value:" << i << std::endl;

    key = 4;
    for (i = 0; i < myvec.size() && myvec[i] < key; i++)
    {
    }
    std::cout << "key:" << key << ", i value:" << i << std::endl;

    key = 5;
    for (i = 0; i < myvec.size() && myvec[i] < key; i++)
    {
    }
    std::cout << "key:" << key << ", i value:" << i << std::endl;

    key = 6;
    for (i = 0; i < myvec.size() && myvec[i] < key; i++)
    {
    }
    std::cout << "key:" << key << ", i value:" << i << std::endl;

    return 0;
}
