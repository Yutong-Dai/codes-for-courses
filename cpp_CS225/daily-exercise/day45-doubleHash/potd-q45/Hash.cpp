#include <vector>
#include "Hash.h"
#include <iostream>
using namespace std;

void doubleHashInput(std::vector<int> &v, int elem)
{
  //your code here
  // v[elem] = -1; //remove this

  int length = v.size();
  int idx = firstHash(elem, length);
  cout << "Input elem: " << elem << "; Attempt to insert at: " << idx << "; Current V size: " << length << endl;
  if (v[idx] == -1)
  {
    cout << "Empty! Insert " << elem << " at idx: " << idx << endl;
    v[idx] = elem;
  }
  else
  {
    int stepSize = secondHash(elem);
    cout << "Not Empty! StepSize: " << stepSize << endl;
    while (idx <= (length - 1))
    {
      idx = (idx + stepSize) % length;
      if (v[idx] == -1)
      {
        cout << "Take one stepSize and Insert " << elem << " at idx: " << idx << endl;
        v[idx] = elem;
        break;
      }
    }
  }
}

//make a hash function called firstHash
int firstHash(int elem, int length)
{
  int idx = (elem * 4) % length;
  return idx;
}

//make a second function called secondHash
int secondHash(int elem)
{
  int stepSize = 3 - elem % 3;
  return stepSize;
}