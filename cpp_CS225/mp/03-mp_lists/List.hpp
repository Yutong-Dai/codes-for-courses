/**
 * @file list.cpp
 * Doubly Linked List (MP 3).
 */
#include<iostream>
template <class T>
List<T>::List() { 
  // @TODO: graded in MP3.1 (done)
    head_ = NULL;
    tail_ = NULL;
    length_ = 0;
}

/**
 * Returns a ListIterator with a position at the beginning of
 * the List.
 */
template <typename T>
typename List<T>::ListIterator List<T>::begin() const {
  // @TODO: graded in MP3.1 (done)
  return List<T>::ListIterator(head_);
}

/**
 * Returns a ListIterator one past the end of the List.
 */
template <typename T>
typename List<T>::ListIterator List<T>::end() const {
  // @TODO: graded in MP3.1 (done)
  return List<T>::ListIterator(NULL);
}


/**
 * Destroys all dynamically allocated memory associated with the current
 * List class.
 */
template <typename T>
void List<T>::_destroy() {
  /// @todo Graded in MP3.1 (done)
  ListNode* current = head_;
  // save space no need to create temp to track previous
  while( current!= NULL){
    current = current->next;
    if(current!= NULL){// take care of the last node case
        delete current->prev;
    }
  }
  if (tail_ != NULL){ // take care of the only one node case
    delete tail_;
  }

  // possibly save run time
  // ListNode* curr = head_;
  // ListNode* temp;
  // // iterate down the parameter List
  // while(curr != NULL){
  //   temp = curr;
  //   curr = curr->next;
  //   delete temp;
  // }
}

/**
 * Inserts a new node at the front of the List.
 * This function **SHOULD** create a new ListNode.
 *
 * @param ndata The data to be inserted.
 */
template <typename T>
void List<T>::insertFront(T const & ndata) {
  /// @todo Graded in MP3.1 (done)
  ListNode * newNode = new ListNode(ndata);
  newNode -> next = head_;
  newNode -> prev = NULL;
  
  if (head_ != NULL) {
    head_ -> prev = newNode;
  }
  if (tail_ == NULL) {
    tail_ = newNode;
  }
  
  head_ = newNode;
  length_++;


}

/**
 * Inserts a new node at the back of the List.
 * This function **SHOULD** create a new ListNode.
 *
 * @param ndata The data to be inserted.
 */
template <typename T>
void List<T>::insertBack(const T & ndata) {
  /// @todo Graded in MP3.1
  ListNode * newNode = new ListNode(ndata);
  newNode -> next = NULL;
  newNode -> prev = tail_;
  
  if (tail_ != NULL) {
    tail_ -> next = newNode;
  }
  if (head_ == NULL) {
    head_ = newNode;
  }
  
  tail_ = newNode;
  length_++;
}

/**
 * Helper function to split a sequence of linked memory at the node
 * splitPoint steps **after** start. In other words, it should disconnect
 * the sequence of linked memory after the given number of nodes, and
 * return a pointer to the starting node of the new sequence of linked
 * memory.
 *
 * This function **SHOULD NOT** create **ANY** new List or ListNode objects!
 *
 * This function is also called by the public split() function located in
 * List-given.hpp
 *
 * @param start The node to start from.
 * @param splitPoint The number of steps to walk before splitting.
 * @return The starting node of the sequence that was split off.
 */
template <typename T>
typename List<T>::ListNode * List<T>::split(ListNode * start, int splitPoint) {
  /// @todo Graded in MP3.1
  if (splitPoint <=0){
    return head_;
  }
  ListNode * curr = start;

  for (int i = 0; i < splitPoint && curr != NULL; i++) {
    curr = curr->next;
  }

  if (curr != NULL) {
      curr->prev->next = NULL;
      curr->prev = NULL;
      return curr;
  }else{
    return head_;
  }
}

/**
 * Modifies the List using the waterfall algorithm.
 * Every other node (starting from the second one) is removed from the
 * List, but appended at the back, becoming the new tail. This continues
 * until the next thing to be removed is either the tail (**not necessarily
 * the original tail!**) or NULL.  You may **NOT** allocate new ListNodes.
 * Note that since the tail should be continuously updated, some nodes will
 * be moved more than once.
 */
template <typename T>
void List<T>::waterfall() {
  /// @todo Graded in MP3.1
  ListNode* temp1 = head_;
  ListNode* temp2 = temp1->next;
  while (temp2 != tail_){
      temp1 -> next = temp2 -> next;
      (temp2 -> next) -> prev = temp1;
      tail_->next = temp2;
      temp2->prev = tail_;
      temp2->next = NULL;
      tail_ = temp2;
      temp1 = temp1->next;
      temp2 = temp1->next;
  }

}


/**
 * Reverses the current List.
 */
template <typename T>
void List<T>::reverse() {
  reverse(head_, tail_);
}

/**
 * Helper function to reverse a sequence of linked memory inside a List,
 * starting at startPoint and ending at endPoint. You are responsible for
 * updating startPoint and endPoint to point to the new starting and ending
 * points of the rearranged sequence of linked memory in question.
 *
 * @param startPoint A pointer reference to the first node in the sequence
 *  to be reversed.
 * @param endPoint A pointer reference to the last node in the sequence to
 *  be reversed.
 */
// template <typename T>
// void List<T>::reverse(ListNode *& startPoint, ListNode *& endPoint) {
//   /// @todo Graded in MP3.2
//   ListNode* startPrev = startPoint->prev;
//   ListNode* endNext = endPoint -> next;
//   ListNode* temp;
//   ListNode* tempPrev;
//   if (endNext == NULL){// endPoint is end
//     // std::cout << "1here" << std::endl;
//     temp = endPoint->prev;
//     tempPrev = endPoint;
//     endPoint->next = temp;
//     endPoint->prev = NULL;
//     if(temp != NULL){
//       while (temp!=startPoint){
//         temp->next = temp->prev;
//         temp->prev = tempPrev;
//         tempPrev = temp;
//         temp = temp->next;
//       }
//       startPoint->next = startPrev;
//       startPoint->prev = tempPrev;
//       head_ = endPoint;
//       if (startPrev==NULL){
//         tail_ = startPoint;
//       }
//     }
//   }else{//endPoint is not end
//     // std::cout << "2here" << std::endl;
//     if (startPrev!=NULL){//startPoint is not head
//       startPrev->next = endPoint;
//     }else{
//       head_ = endPoint;
//     }
//     if (endPoint->prev!=NULL){
//       temp = endPoint->prev;
//       tempPrev = endPoint;
//       while (temp!=startPoint){
//         temp->next = temp->prev;
//         temp->prev = tempPrev;
//         tempPrev = temp;
//         temp = temp->next;
//       }
//       startPoint->next = endPoint->next;
//     } 
//   }
// }
template <typename T>
void List<T>::reverse(ListNode *& startPoint, ListNode *& endPoint) {
  // @todo Graded in MP3.2
  if((startPoint != endPoint) && (startPoint != NULL) && (endPoint != NULL)){
    ListNode* temp1;
    ListNode* temp2;
    ListNode* cur_start;
    ListNode* cur_end;

    if(startPoint->prev != NULL){
      (startPoint->prev)->next = endPoint;
    }

    if(endPoint->next != NULL){
      (endPoint->next)->prev = startPoint;
    }

    if(startPoint->next == endPoint){
      // this means end is next to start
      startPoint->next = endPoint->next;
      endPoint->prev = startPoint->prev;
      startPoint->prev = endPoint;
      endPoint->next = startPoint;
      // Modify the startPointer and endPointer
      // temp1 = startPoint;
      // startPoint = endPoint;
      // endPoint = temp1;
      return;
    }
    else{
      temp1 = startPoint->next;
      temp2 = endPoint->prev;
      startPoint->next = endPoint->next;
      endPoint->prev = startPoint->prev;
      startPoint->prev = temp2;
      endPoint->next = temp1;
      temp1->prev = endPoint;
      temp2->next = startPoint;
      // Modify the startPointer and endPointer
      temp1 = startPoint;
      startPoint = endPoint;
      endPoint = temp1;
    }

    cur_start = startPoint->next;
    cur_end = endPoint->prev;

    while(!((cur_start == cur_end) || (cur_start->next == cur_end))){
      (cur_start->prev)->next = cur_end;
      (cur_end->next)->prev = cur_start;
      temp1 = cur_start->next;
      temp2 = cur_end->prev;

      cur_start->next = cur_end->next;
      cur_end->prev = cur_start->prev;

      cur_start->prev = temp2;
      cur_end->next = temp1;

      temp1->prev = cur_end;
      temp2->next = cur_start;
      //Now the current start and current end is exchanged

      cur_start = temp1;
      cur_end = temp2;
    }
    // (cur_start == cur_end) || (cur_start->next == cur_end)
    // If there are odd nodes, then the loop should stop when cur_start==cur_end.
    // After that, all nodes are reversed. Don't need to do anything else.
    // If there are even nodes, the loop should stop when cur_start->next == cur_end
    // Due to the way I use temp1 & temp2, I must do one more step to exchange the last two nodes.

    if(cur_start->next == cur_end){
      // this means cur_end is next to cur_start
      (cur_start->prev)->next = cur_end;
      (cur_end->next)->prev = cur_start;

      cur_start->next = cur_end->next;
      cur_end->prev = cur_start->prev;

      cur_start->prev = cur_end;
      cur_end->next = cur_start;
    }
  }
}
  

/**
 * Reverses blocks of size n in the current List. You should use your
 * reverse( ListNode * &, ListNode * & ) helper function in this method!
 *
 * @param n The size of the blocks in the List to be reversed.
 */
template <typename T>
void List<T>::reverseNth(int n) {
  /// @todo Graded in MP3.2
  if(n >= length_){
    reverse(head_, tail_);
    return;
  }
  else{
    ListNode* startPointer = head_;
    ListNode* endPointer = head_;

    // Reverse the first block (Do this separately so that head_ is modified correctly)
    // first find the endPointer
    for(int j = 1; j < n; j++){
      endPointer = endPointer->next;
    }
    reverse(head_, endPointer);

    for(int i=2; i*n < length_; i++){
      startPointer = endPointer->next;
      for(int j = 0; j < n; j++){
        endPointer = endPointer->next;
      }
      reverse(startPointer, endPointer);
    }

    startPointer = endPointer->next;
    reverse(startPointer, tail_);
  }
}


/**
 * Merges the given sorted list into the current sorted list.
 *
 * @param otherList List to be merged into the current list.
 */
template <typename T>
void List<T>::mergeWith(List<T> & otherList) {
    // set up the current list
    head_ = merge(head_, otherList.head_);
    tail_ = head_;

    // make sure there is a node in the new list
    if (tail_ != NULL) {
        while (tail_->next != NULL)
            tail_ = tail_->next;
    }
    length_ = length_ + otherList.length_;

    // empty out the parameter list
    otherList.head_ = NULL;
    otherList.tail_ = NULL;
    otherList.length_ = 0;
}

/**
 * Helper function to merge two **sorted** and **independent** sequences of
 * linked memory. The result should be a single sequence that is itself
 * sorted.
 *
 * This function **SHOULD NOT** create **ANY** new List objects.
 *
 * @param first The starting node of the first sequence.
 * @param second The starting node of the second sequence.
 * @return The starting node of the resulting, sorted sequence.
 */
template <typename T>
typename List<T>::ListNode * List<T>::merge(ListNode * first, ListNode* second) {
  /// @todo Graded in MP3.2
  if(first == NULL && second == NULL){
    return NULL;
  }
  else if(first == NULL){
    return second;
  }
  else if(second == NULL){
    return first;
  }

  ListNode* temp1;
  ListNode* temp2;
  ListNode* head;
  if(first->data < second->data){
    head = first;
    temp1 = first;
    temp2 = second;
  }
  else{
    head = second;
    temp1 = second;
    temp2 = first;
  }

  while(true){
    if(((temp1->next) != NULL) && ((temp2->next) != NULL)){
      // While both temp1 and temp2 do NOT reach the end, keep looping
      if(temp2->data < (temp1->next)->data){
        // After last round of loop, there must be temp1->data <= temp2->data
        // So only need to check if temp2->data < (temp1->next)->data
        (temp1->next)->prev = temp2;
        temp2->prev = temp1;
        temp2 = temp2->next;   // if temp2 reaches the end, then this step is redundant.
                      // Since now don't need to keep track of the next element in temp2
        (temp2->prev)->next = temp1->next;
        temp1->next = temp2->prev;
      }
      temp1 = temp1->next;
    }
    else if(temp1->next != NULL){
      // Reaching this step means, at least one of temp1 and temp2 reaches the end
      // if temp1 does NOT reaches the end, it means temp2 reaches the end.(there must be temp2->next == NULL)
      // keep looping to find the right place for temp2
      if(temp2->data < (temp1->next)->data){
        (temp1->next)->prev = temp2;
        temp2->next = temp1->next;
        temp2->prev = temp1;
        temp1->next = temp2;
        break;
      }
      temp1 = temp1->next;
    }
    else{
      // There are two possible paths to reach this step
      // 1. temp2 already reaches end, keep looping on temp1, finally temp1 also reaches the end
      // 2. temp1 reaches the end, there may or may not be remaining elements in temp2
      // either way, if there are remaining elements in temp2, they must be larger than temp1
      // so just attach temp2 behind temp1
      temp1->next = temp2;
      temp2->prev = temp1;
      break;
    }
  }

  return head;
}
/**
 * Sorts a chain of linked memory given a start node and a size.
 * This is the recursive helper for the Mergesort algorithm (i.e., this is
 * the divide-and-conquer step).
 *
 * Called by the public sort function in List-given.hpp
 *
 * @param start Starting point of the chain.
 * @param chainLength Size of the chain to be sorted.
 * @return A pointer to the beginning of the now sorted chain.
 */
template <typename T>
typename List<T>::ListNode* List<T>::mergesort(ListNode * start, int chainLength) {
  /// @todo Graded in MP3.2
  if(start == NULL){
    return NULL;
  }
  else if(chainLength == 1){
    return start;
  }
  ListNode* splitPoint;
  int length1, length2;

  length1 = chainLength / 2;           // length of first half
  length2 = chainLength - length1;     // length of second half
  splitPoint = split(start, length1);  // startPoint of second half
  return merge(mergesort(start, length1), mergesort(splitPoint, length2));
  //return merge(mergesort(start, chainLength/2), mergesort(split(start, chainLength/2), chainLength-chainLength/2));
}
