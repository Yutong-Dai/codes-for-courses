#include <iostream>
// problem 1
// out-of-bounds access to heap
// 1. should change 100 to 99. But this will cause another warining, returning unintialized int
// didn't free the heap memory; delete[] arr;
// 
// int main(){
//     int * arr = new int[100];
//     return arr[100];
// }

// problem 2
// use of an unintialized value
// int main(){
//     int x;
//     std::cout << x << std::endl;
// }