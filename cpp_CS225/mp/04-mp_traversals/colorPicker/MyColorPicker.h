#pragma once

#include "ColorPicker.h"
#include "../cs225/HSLAPixel.h"
#include "../Point.h"
#include "../cs225/PNG.h"

using namespace cs225;

/**
 * A color picker class using your own color picking algorithm
 */
class MyColorPicker : public ColorPicker
{
public:
  HSLAPixel getColor(unsigned x, unsigned y);
  MyColorPicker(PNG &png_);

private:
  PNG png;
  unsigned int width_;
  unsigned int height_;
};
