#include <vector>
#include "Primes.h"
#include <iostream>
using namespace std;
std::vector<int> *genPrimes(int M)
{
    std::vector<int> *v = new std::vector<int>();
    std::vector<int> *t = new std::vector<int>(M);
    int i = 0;
    int nextPrime = 2;

    for (i = 0; i < M; i++)
        (*t)[i] = 1;

    (*t)[0] = 0;
    (*t)[1] = 0;

    v->push_back(2);

    while (nextPrime < M)
    {
        for (i = nextPrime * nextPrime;
             i < M;
             i += nextPrime)
            (*t)[i] = 0;
        for (++nextPrime; nextPrime < M; nextPrime++)
            if ((*t)[nextPrime] == 1)
            {
                v->push_back(nextPrime);
                break;
            }
    }

    delete t;
    return v;
}

int numSequences_(std::vector<int>::iterator it, int num, std::vector<int> *primes)
{
    int gap = num - *it;
    // cout << num << "|" << *it << "|" << gap << "|" << *(it - 1) << endl;

    while (it != primes->begin() and *(it - 1) != 0)
    {
        // cout << "Current pointer: " << *it << " Current Gap:" << gap << endl;
        if (gap >= (*(it - 1)))
        {
            it -= 1;
            gap -= *it;
            if (gap == 0)
            {
                // cout << "done: return 1" << endl;
                return 1;
            }
        }
        else
        {
            // cout << "gap < adjacent, return 0" << endl;
            return 0;
        }
    }
    // cout << "Out of while, Current pointer: " << *it << " Current Gap:" << gap << endl;
    if (gap == 0)
    {
        // cout << "done till the begin: return 1" << endl;
        return 1;
    }
    else
    {
        // cout << "return 0" << endl;
        return 0;
    }
}
int numSequences(std::vector<int> *primes, int num)
{

    // your code here
    int numSeq = 0;
    std::vector<int>::iterator it = primes->begin();
    while (it != primes->end())
    {
        if (*it == num)
        {
            numSeq += 1;
            break;
        }
        it++;
    }
    // if (numSeq >= 1)
    // {
    while (it != primes->begin())
    {
        it -= 1; // the prime number that is closest to the input num
        numSeq += numSequences_(it, num, primes);
        // cout << "====" << endl;
    }
    // }
    std::cout << num << " has " << numSeq << " sequence(s)." << std::endl;
    return numSeq;
}