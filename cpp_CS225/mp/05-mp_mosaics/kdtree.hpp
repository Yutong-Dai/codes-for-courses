/**
 * @file kdtree.cpp
 * Implementation of KDTree class.
 */

#include <utility>
#include <algorithm>

using namespace std;

template <int Dim>
bool KDTree<Dim>::smallerDimVal(const Point<Dim> &first,
                                const Point<Dim> &second, int curDim) const
{
  /**
     * @todo Implement this function!
     */
  if (first[curDim] == second[curDim])
  {
    return first < second;
  }
  else
  {
    return first[curDim] < second[curDim];
  }
}

template <int Dim>
bool KDTree<Dim>::shouldReplace(const Point<Dim> &target,
                                const Point<Dim> &currentBest,
                                const Point<Dim> &potential) const
{
  /**
     * @todo Implement this function!
     */
  double curDist = 0;
  double potDist = 0;
  for (int i = 0; i < Dim; i++)
  {
    curDist += (target[i] - currentBest[i]) * (target[i] - currentBest[i]);
    potDist += (target[i] - potential[i]) * (target[i] - potential[i]);
  }

  if (curDist == potDist)
  {
    return potential < currentBest;
  }
  else
  {
    return potDist < curDist;
  }

  return false;
}

template <int Dim>
KDTree<Dim>::KDTree(const vector<Point<Dim>> &newPoints)
{
  /**
     * @todo Implement this function!
     */
  if (newPoints.size() == 0)
  {
    // The case that the vector has no Point in it.
    root = NULL;
    size = 0;
  }
  else
  {
    size = newPoints.size();
    vector<Point<Dim>> sortedPoints;
    sortedPoints.assign(newPoints.begin(), newPoints.end());
    root = kdtreeBuilder(sortedPoints, 0, size - 1, 0);
  }
}
template <int Dim>
typename KDTree<Dim>::KDTreeNode *KDTree<Dim>::kdtreeBuilder(vector<Point<Dim>> &points, int curBegin, int curEnd, int curDim)
{
  if (curBegin == curEnd)
  {
    KDTreeNode *curRoot = new KDTreeNode(points[curBegin]);
    return curRoot;
  }
  int midIndex = (curBegin + curEnd) / 2;
  Point<Dim> curMedian = quickSelect(points, curBegin, curEnd, midIndex, curDim);
  KDTreeNode *curRoot = new KDTreeNode(curMedian);
  if (curBegin < midIndex)
  {
    // when there are only 2 elements between left and right, curBegin == midIndex
    curRoot->left = kdtreeBuilder(points, curBegin, midIndex - 1, (curDim + 1) % Dim);
  }
  curRoot->right = kdtreeBuilder(points, midIndex + 1, curEnd, (curDim + 1) % Dim);
  return curRoot;
}

template <int Dim>
int KDTree<Dim>::partition(vector<Point<Dim>> &points, int curBegin, int curEnd, int pivotIndex, int curDim)
{
  Point<Dim> pivotPoint = points[pivotIndex];
  //double pivotValue = pivotPoint[curDim];
  points[pivotIndex] = points[curEnd];

  int storeIndex = curBegin;
  Point<Dim> temp;
  for (int i = curBegin; i < curEnd; i++)
  {
    if (points[i][curDim] < pivotPoint[curDim] || ((points[i][curDim] == pivotPoint[curDim]) && (points[i] < pivotPoint)))
    {
      temp = points[i];
      points[i] = points[storeIndex];
      points[storeIndex] = temp;
      storeIndex++;
    }
  }

  points[curEnd] = points[storeIndex];
  points[storeIndex] = pivotPoint;
  return storeIndex;
}

template <int Dim>
Point<Dim> KDTree<Dim>::quickSelect(vector<Point<Dim>> &points, int curBegin, int curEnd, int target, int curDim)
{
  int pivotIndex = (curBegin + curEnd) / 2;
  int idx = partition(points, curBegin, curEnd, pivotIndex, curDim);
  if (idx == target)
  {
    return points[idx];
  }
  else if (idx > target)
  {
    return quickSelect(points, curBegin, idx - 1, target, curDim);
  }
  else
  {
    return quickSelect(points, idx + 1, curEnd, target, curDim);
  }
}

template <int Dim>
KDTree<Dim>::KDTree(const KDTree<Dim> &other)
{
  /**
   * @todo Implement this function!
   */
  root = other.root;
  size = other.size;
}

template <int Dim>
const KDTree<Dim> &KDTree<Dim>::operator=(const KDTree<Dim> &rhs)
{
  /**
   * @todo Implement this function!
   */
  root = rhs.root;
  size = rhs.size;
  return *this;
}

template <int Dim>
KDTree<Dim>::~KDTree()
{
  /**
   * @todo Implement this function!
   */
  kdtreeDelete(root);
}
template <int Dim>
void KDTree<Dim>::kdtreeDelete(KDTreeNode *curNode)
{
  if (curNode == NULL)
  {
    return;
  }

  kdtreeDelete(curNode->left);
  kdtreeDelete(curNode->right);
  delete curNode;
  curNode = NULL;
}

template <int Dim>
Point<Dim> KDTree<Dim>::findNearestNeighbor(const Point<Dim> &query) const
{
  /**
     * @todo Implement this function!
     */

  return findNearestNeighbor(query, root, 0);
}

template <int Dim>
Point<Dim> KDTree<Dim>::findNearestNeighbor(const Point<Dim> &query, KDTreeNode *curNode, int curDim) const
{
  Point<Dim> curPoint = curNode->point;

  if (curNode->left == NULL && curNode->right == NULL)
  {
    // base case : reach leaf node
    return curPoint;
  }

  // if current node is not a leaf node
  Point<Dim> curBest;

  if (smallerDimVal(query, curPoint, curDim))
  {
    // query is smaller than curPoint, should go left
    if (curNode->left == NULL)
    {
      curBest = curPoint;
    }
    else
    {
      curBest = findNearestNeighbor(query, curNode->left, (curDim + 1) % Dim);
      if (shouldReplace(query, curBest, curPoint))
      {
        curBest = curPoint;
      }
    }

    // calculate current radius
    double curDist = 0;
    for (int i = 0; i < Dim; i++)
    {
      curDist += (query[i] - curBest[i]) * (query[i] - curBest[i]);
    }

    // Note!!!
    // here when <=, should check another subtree!
    // (Although distance in another subtree may be the same, can find another point < current best)
    if ((curPoint[curDim] - query[curDim]) * (curPoint[curDim] - query[curDim]) <= curDist && curNode->right != NULL)
    {
      Point<Dim> potential = findNearestNeighbor(query, curNode->right, (curDim + 1) % Dim);
      if (shouldReplace(query, curBest, potential))
      {
        curBest = potential;
      }
    }
  }
  else
  {
    // should go right
    if (curNode->right == NULL)
    {
      curBest = curPoint;
    }
    else
    {
      curBest = findNearestNeighbor(query, curNode->right, (curDim + 1) % Dim);
      if (shouldReplace(query, curBest, curPoint))
      {
        curBest = curPoint;
      }
    }

    // calculate current radius
    double curDist = 0;
    for (int i = 0; i < Dim; i++)
    {
      curDist += (query[i] - curBest[i]) * (query[i] - curBest[i]);
    }

    // Note!!!
    // here when <=, should check another subtree!
    // (Although distance in another subtree may be the same, can find another point < current best)
    if ((curPoint[curDim] - query[curDim]) * (curPoint[curDim] - query[curDim]) <= curDist && curNode->left != NULL)
    {
      Point<Dim> potential = findNearestNeighbor(query, curNode->left, (curDim + 1) % Dim);
      if (shouldReplace(query, curBest, potential))
      {
        curBest = potential;
      }
    }
  }
  return curBest;
}
