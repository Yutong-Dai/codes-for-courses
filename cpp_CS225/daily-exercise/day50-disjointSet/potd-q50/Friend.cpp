#include "Friend.h"
#include <algorithm>
#include <iostream>
// int find(int x, std::vector<int> &parents)
// {
//    return parents[x];
// }

int findCircleNum(std::vector<std::vector<int>> &M)
{
   // your code
   unsigned int Msize = M.size();
   unsigned int CircleNum = Msize;
   std::vector<int> parents;

   for (unsigned int k = 0; k < Msize; k++)
   {
      parents.push_back(k);
   }

   for (unsigned int i = 1; i <= Msize - 1; i++)
   {
      std::vector<int> currentRow = M[i];
      std::vector<int> tempParents;
      for (unsigned int j = 0; j <= i - 1; j++)
      {
         if (currentRow[j] == 1)
         {
            tempParents.push_back(parents[j]);
            CircleNum -= 1;
         }
      }
      // std::cout << "size of tempParents: " << tempParents.size() << std::endl;
      if (tempParents.size() >= 2)
      {
         int minParent = tempParents.front();
         for (unsigned int k = 0; k <= i - 1; k++)
         {
            // std::cout << parents[k] << std::endl;
            std::vector<int>::iterator it = find(tempParents.begin(), tempParents.end(), parents[k]);
            if (it != tempParents.end())
            {
               // std::cout << "catch! Substitue with: " << minParent << std::endl;
               parents[k] = minParent;
            }
         }
         parents[i] = minParent;
      }
      else
      {
         if (!tempParents.empty())
         {
            parents[i] = tempParents.front();
         }
      }
   }

   std::cout << "Parent index: ";
   for (std::vector<int>::iterator it = parents.begin(); it != parents.end(); it++)
   {
      std::cout << *it << "|";
   }
   std::cout << std::endl;
   return CircleNum;
}
