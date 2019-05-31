#include "../cs225/HSLAPixel.h"
#include "../Point.h"

#include "ColorPicker.h"
#include "MyColorPicker.h"

using namespace cs225;

/**
 * Picks the color for pixel (x, y).
 * Using your own algorithm
 */

MyColorPicker::MyColorPicker(PNG &png_) : png(png_)
{
  width_ = png.width();
  height_ = png.height();
}

HSLAPixel MyColorPicker::getColor(unsigned x, unsigned y)
{
  /* @todo [Part 3] */
  HSLAPixel pixel(0, 0, 1, 0);
  if (x < width_ && y < height_)
  {
    pixel = png.getPixel(x, y);
  }
  return pixel;
}
