#include <vector>
#include "Primes.h"

std::vector<int> *genPrimes(int M) {
    std::vector<int> *v = new std::vector<int>();
    // your code here
    std::vector<bool> marked(M-1, false);  // 2 ~ M, M-1 numbers
    for(int p = 2; p <= M; p++){
      if(!marked[p-2]){
        v->push_back(p);
        for(int k = 2; k <= M/p; k++){
          marked[k*p-2] = true;
        }
      }
    }
    return v;
}
