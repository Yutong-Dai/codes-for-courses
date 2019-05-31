#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include "BTreeNode.h"

int main()
{
  std::vector<int> v1{30, 60};
  std::vector<int> v2{10, 20};
  std::vector<int> v3{40, 50};
  std::vector<int> v4{70, 80};

  BTreeNode n_0(v1), n_2(v2), n_3(v3), n_4(v4);
  n_0.children_.push_back(&n_2);
  n_0.children_.push_back(&n_3);
  n_0.children_.push_back(&n_4);
  n_0.is_leaf_ = false;

  n_2.is_leaf_ = false;
  n_3.is_leaf_ = false;
  n_4.is_leaf_ = false;

  std::vector<int> v8{8, 9};
  std::vector<int> v9{38, 39};
  std::vector<int> v10{68, 69};
  BTreeNode n_8(v8), n_9(v9), n_10(v10);
  n_2.children_.push_back(&n_8);
  n_3.children_.push_back(&n_9);
  n_4.children_.push_back(&n_10);

  std::vector<int> v5{11, 12};
  std::vector<int> v6{41, 42};
  std::vector<int> v7{71, 72};
  BTreeNode n_5(v5), n_6(v6), n_7(v7);
  n_2.children_.push_back(&n_5);
  n_3.children_.push_back(&n_6);
  n_4.children_.push_back(&n_7);

  std::vector<int> v11{21, 22};
  std::vector<int> v12{51, 52};
  std::vector<int> v13{81, 82};
  BTreeNode n_11(v11), n_12(v12), n_13(v13);
  n_2.children_.push_back(&n_11);
  n_3.children_.push_back(&n_12);
  n_4.children_.push_back(&n_13);

  std::vector<int> tr = traverse(&n_0);
  for (std::vector<int>::const_iterator i = tr.begin(); i != tr.end(); ++i)
    std::cout << *i << ' ';
  std::cout << std::endl;

  return 0;
}
