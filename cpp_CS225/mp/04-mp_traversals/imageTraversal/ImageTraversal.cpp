#include <cmath>
#include <iterator>
#include <iostream>

#include "../cs225/HSLAPixel.h"
#include "../cs225/PNG.h"
#include "../Point.h"

#include "ImageTraversal.h"

/**
 * Calculates a metric for the difference between two pixels, used to
 * calculate if a pixel is within a tolerance.
 * 
 * @param p1 First pixel
 * @param p2 Second pixel
 * @return the difference between two HSLAPixels
 */
double ImageTraversal::calculateDelta(const HSLAPixel &p1, const HSLAPixel &p2)
{
  double h = fabs(p1.h - p2.h);
  double s = p1.s - p2.s;
  double l = p1.l - p2.l;

  // Handle the case where we found the bigger angle between two hues:
  if (h > 180)
  {
    h = 360 - h;
  }
  h /= 360;

  return sqrt((h * h) + (s * s) + (l * l));
}

/**
 * Default iterator constructor.
 */
ImageTraversal::Iterator::Iterator()
{
  /** @todo [Part 1] */
  traverseMethod_ = NULL;
  /*nothing need to be done.*/
}

//
ImageTraversal::Iterator::Iterator(ImageTraversal *traverseMethod, PNG &basepng, Point startPoint, double tolerance)
{
  /** @todo [Part 1] */
  traverseMethod_ = traverseMethod;
  basepng_ = basepng;
  width_ = basepng.width();
  height_ = basepng.height();
  startPoint_ = startPoint;
  startPixel_ = basepng_.getPixel(startPoint_.x, startPoint_.y);
  tolerance_ = tolerance;

  currentPoint_ = startPoint_;
  std::cout << " ======= INITIALIZE =======" << std::endl;
  std::cout << " Now, we are at" << std::endl;
  std::cout << currentPoint_ << std::endl;

  visitedMat_.resize(width_ * height_, 0);
  visitedMat_[currentPoint_.x + currentPoint_.y * width_] = 1;
  std::cout << "Then, we visit it." << std::endl;
  std::cout << "The top of the current stack" << std::endl;
  std::cout << traverseMethod_->peek() << std::endl;
}

/**
 * Iterator increment opreator.
 *
 * Advances the traversal of the image.
 */
ImageTraversal::Iterator &ImageTraversal::Iterator::operator++()
{
  /** @todo [Part 1] */
  // std::cout << std::boolalpha << traverseMethod_->empty() << std::endl;

  if (!traverseMethod_->empty())
  {
    std::cout << "=========inside one ++ operation == " << std::endl;
    std::cout << "remove current top of the stack" << std::endl;
    traverseMethod_->pop();
    Point right = Point(currentPoint_.x + 1, currentPoint_.y);
    Point below = Point(currentPoint_.x, currentPoint_.y + 1);
    Point left = Point(currentPoint_.x - 1, currentPoint_.y);
    Point above = Point(currentPoint_.x, currentPoint_.y - 1);
    if ((right.x < width_) && (visitedMat_[right.x + right.y * width_] == 0) && (calculateDelta(basepng_.getPixel(right.x, right.y), startPixel_) < tolerance_))
    {
      traverseMethod_->add(right);
      std::cout << "Add Right" << std::endl;
      std::cout << right << std::endl;
    }
    if ((below.y < height_) && (visitedMat_[below.x + below.y * width_] == 0) && (calculateDelta(basepng_.getPixel(below.x, below.y), startPixel_) < tolerance_))
    {
      traverseMethod_->add(below);
      std::cout << "Add Below" << std::endl;
      std::cout << below << std::endl;
    }
    if ((currentPoint_.x >= 1) && (visitedMat_[left.x + left.y * width_] == 0) && (calculateDelta(basepng_.getPixel(left.x, left.y), startPixel_) < tolerance_))
    {
      traverseMethod_->add(left);
      std::cout << "Add Left" << std::endl;
      std::cout << left << std::endl;
    }
    if ((currentPoint_.y >= 1) && (visitedMat_[above.x + above.y * width_] == 0) && (calculateDelta(basepng_.getPixel(above.x, above.y), startPixel_) < tolerance_))
    {
      traverseMethod_->add(above);
      std::cout << "Add Above" << std::endl;
      std::cout << above << std::endl;
    }
    std::cout << "Finishing adding" << std::endl;
    if (!traverseMethod_->empty())
    {
      std::cout << "The top of the modified stack" << std::endl;
      std::cout << traverseMethod_->peek() << std::endl;
    }
    else
    {
      std::cout << "Done!" << std::endl;
    }

    while (!traverseMethod_->empty()) // has to use while to iteratively delete the duplicates
    {
      std::cout << "We move according to the top of the stack." << std::endl;
      currentPoint_ = traverseMethod_->peek();
      if (visitedMat_[currentPoint_.x + currentPoint_.y * width_] == 0) // if not visited, move
      {
        std::cout << "Now, we are at" << std::endl;
        std::cout << currentPoint_ << std::endl;
        std::cout << "Then, we visit it. " << std::endl;
        visitedMat_[currentPoint_.x + currentPoint_.y * width_] = 1;
        break;
      }
      else // if visited, remove the duplicate from the neighobr_
      {
        std::cout << "Opps, can not move. Remove the duplicate Instead." << std::endl;
        Point duplicate = traverseMethod_->pop();
        std::cout << duplicate << std::endl;
      }
    }
  }

  return *this; // return updated iterator
}

/**
 * Iterator accessor opreator.
 * 
 * Accesses the current Point in the ImageTraversal.
 */
Point ImageTraversal::Iterator::operator*()
{
  /** @todo [Part 1] */
  return this->currentPoint_;
}

/**
 * Iterator inequality operator.
 * 
 * Determines if two iterators are not equal.
 */
bool ImageTraversal::Iterator::operator!=(const ImageTraversal::Iterator &other)
{
  /** @todo [Part 1] */
  // compare with the one past the object, i.e. null objetc.
  bool thisEmpty;
  bool otherEmpty;

  if (traverseMethod_ == NULL)
  {
    thisEmpty = true;
  }
  else
  {
    thisEmpty = traverseMethod_->empty();
  }
  if (other.traverseMethod_ == NULL)
  {
    otherEmpty = true;
  }
  else
  {
    otherEmpty = other.traverseMethod_->empty();
  }

  if (thisEmpty && otherEmpty)
  {
    std::cout << "the same as the iterator.end(), stop!" << std::endl;
    return false;
  } // both empty then the traversals are equal, return true
  else if ((!thisEmpty) && (!otherEmpty))
  {
    std::cout << "can't happen in the use case" << std::endl;
    return (traverseMethod_ != other.traverseMethod_);
  } //both not empty then compare the traversals
  else
  {
    std::cout << "not the same as the iterator.end(), continue!" << std::endl;
    return true;
  } // one is empty while the other is not, return true
}
