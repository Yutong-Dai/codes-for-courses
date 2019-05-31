
/**
 * @file heap.cpp
 * Implementation of a heap class.
 */

template <class T, class Compare>
size_t heap<T, Compare>::root() const
{
    // @TODO Update to return the index you are choosing to be your root.
    // 1-index-root
    return 1;
}

template <class T, class Compare>
size_t heap<T, Compare>::leftChild(size_t currentIdx) const
{
    // @TODO Update to return the index of the left child.
    return currentIdx * 2;
}

template <class T, class Compare>
size_t heap<T, Compare>::rightChild(size_t currentIdx) const
{
    // @TODO Update to return the index of the right child.
    return (currentIdx * 2 + 1);
}

template <class T, class Compare>
size_t heap<T, Compare>::parent(size_t currentIdx) const
{
    // @TODO Update to return the index of the parent.
    return currentIdx / 2;
}

template <class T, class Compare>
bool heap<T, Compare>::hasAChild(size_t currentIdx) const
{
    // @TODO Update to return whether the given node has a child
    if (leftChild(currentIdx) < _elems.size())
    {
        return true;
    }
    return false;
}

template <class T, class Compare>
size_t heap<T, Compare>::maxPriorityChild(size_t currentIdx) const
{
    // @TODO Update to return the index of the child with highest priority
    ///   as defined by higherPriority()
    size_t rightChildIdx = rightChild(currentIdx);
    size_t leftChildIdx = leftChild(currentIdx);
    if (rightChildIdx >= _elems.size() || higherPriority(_elems[leftChildIdx], _elems[rightChildIdx]))
    {
        return leftChildIdx;
    }
    else
    {
        return rightChildIdx;
    }
}

template <class T, class Compare>
void heap<T, Compare>::heapifyDown(size_t currentIdx)
{
    // @TODO Implement the heapifyDown algorithm.

    // current node is NOT a leaf
    if (hasAChild(currentIdx))
    {
        size_t minChildIdx = maxPriorityChild(currentIdx);
        if (higherPriority(_elems[minChildIdx], _elems[currentIdx]))
        {
            std::swap(_elems[minChildIdx], _elems[currentIdx]);
            heapifyDown(minChildIdx);
        }
    }
}

template <class T, class Compare>
void heap<T, Compare>::heapifyUp(size_t currentIdx)
{
    if (currentIdx == root())
        return;
    size_t parentIdx = parent(currentIdx);
    if (higherPriority(_elems[currentIdx], _elems[parentIdx]))
    {
        std::swap(_elems[currentIdx], _elems[parentIdx]);
        heapifyUp(parentIdx);
    }
}

template <class T, class Compare>
heap<T, Compare>::heap()
{
    // @TODO Depending on your implementation, this function may or may
    ///   not need modifying
    _elems.push_back(T());
}

template <class T, class Compare>
heap<T, Compare>::heap(const std::vector<T> &elems)
{
    // @TODO Construct a heap using the buildHeap algorithm
    _elems.assign(elems.begin(), elems.end());
    _elems.insert(_elems.begin(), T());
    for (unsigned i = parent(_elems.size()); i > 0; i--)
    {
        heapifyDown(i);
    }
}

template <class T, class Compare>
T heap<T, Compare>::pop()
{
    // @TODO Remove, and return, the element with highest priority
    T minElem = _elems[root()];
    _elems[root()] = _elems.back();
    _elems.pop_back();
    heapifyDown(root());
    return minElem;
}

template <class T, class Compare>
T heap<T, Compare>::peek() const
{
    // @TODO Return, but do not remove, the element with highest priority
    return _elems[root()];
}

template <class T, class Compare>
void heap<T, Compare>::push(const T &elem)
{
    // @TODO Add elem to the heap
    _elems.push_back(elem);
    heapifyUp(_elems.size() - 1);
}

template <class T, class Compare>
void heap<T, Compare>::updateElem(const size_t &idx, const T &elem)
{
    // @TODO In-place updates the value stored in the heap array at idx
    // Corrects the heap to remain as a valid heap even after update
    T prevElem = _elems[idx];
    _elems[idx] = elem;
    if (higherPriority(elem, prevElem))
    {
        heapifyUp(idx);
    }
    else
    {
        heapifyDown(idx);
    }
}

template <class T, class Compare>
bool heap<T, Compare>::empty() const
{
    // @TODO Determine if the heap is empty
    return true;
}

template <class T, class Compare>
void heap<T, Compare>::getElems(std::vector<T> &heaped) const
{
    for (size_t i = root(); i < _elems.size(); i++)
    {
        heaped.push_back(_elems[i]);
    }
}
