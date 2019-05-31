#include <iostream>  // std::cout
#include <algorithm> // std::fill

int main()
{
    const int nrow = 3;
    const int ncol = 4;
    int mat[nrow][ncol];
    std::fill(mat[0], mat[0] + nrow * ncol, 0);
    std::cout << mat[0][0] << std::endl;
    mat[0][0] = 100;
    std::cout << mat[0][0] << std::endl;
    return 0;
}
