// #include <iostream>
// #include <vector>

// template <typename T>
// std::ostream &operator<<(std::ostream &out, const std::vector<T> &v)
// {
//     if (!v.empty())
//     {
//         out << '[';
//         std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
//         out << "\b\b]";
//     }
//     return out;
// }

// int main()
// {
//     std::vector<int> myvector(1, 1);
//     std::vector<int> copyv;
//     std::vector<int>::iterator it;
//     it = myvector.begin();
//     myvector.insert(it, 2);
//     it = myvector.begin();
//     myvector.insert(it + 1, 3);
//     it = myvector.begin();
//     myvector.insert(it + 1, 4);
//     copyv.assign(myvector.begin(), myvector.end() - 1);
//     std::cout
//         << myvector << std::endl;
//     std::cout
//         << copyv << std::endl;
//     myvector.erase(myvector.begin() + 2);
//     std::cout
//         << myvector << std::endl;
//     myvector.resize(1);
//     std::cout
//         << myvector << std::endl;
//     return 0;
// }
#include <vector>
#include <iostream>
using namespace std;

int main()
{
    vector<int> v, z;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);

    cout
        << "Size of first: " << int(v.size()) << '\n';

    cout << "Elements are\n";
    for (int i = 0; i < v.size(); i++)
        cout << v[i] << endl;

    // modify the elements
    z.assign(v.begin() + 1, v.begin() + 3);

    cout << "\nModified VectorElements are\n";
    for (int i = 0; i < v.size(); i++)
        cout << z[i] << endl;

    z.assign(v.begin() + 1, v.begin() + 4);

    cout << "\nModified VectorElements are\n";
    for (int i = 0; i < v.size(); i++)
        cout << z[i] << endl;
    return 0;
}