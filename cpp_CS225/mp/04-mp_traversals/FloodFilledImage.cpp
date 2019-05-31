#include "cs225/PNG.h"
#include <list>
#include <iostream>

#include "colorPicker/ColorPicker.h"
#include "imageTraversal/ImageTraversal.h"

#include "Point.h"
#include "Animation.h"
#include "FloodFilledImage.h"

using namespace cs225;

/**
 * Constructs a new instance of a FloodFilledImage with a image `png`.
 * 
 * @param png The starting image of a FloodFilledImage
 */
FloodFilledImage::FloodFilledImage(const PNG &png)
{
  /** @todo [Part 2] */
  png_ = png;
}

/**
 * Adds a flood fill operation to the FloodFillImage.  This function must store the operation,
 * which will be used by `animate`.
 * 
 * @param traversal ImageTraversal used for this FloodFill operation.
 * @param colorPicker ColorPicker used for this FloodFill operation.
 */

// the function below is used to traverse the image multiple times
void FloodFilledImage::addFloodFill(ImageTraversal &traversal, ColorPicker &colorPicker)
{
  /** @todo [Part 2] */
  traversals.push_back(&traversal);
  colorpickers.push_back(&colorPicker);
}

/**
 * Creates an Animation of frames from the FloodFill operations added to this object.
 * 
 * Each FloodFill operation added by `addFloodFill` is executed based on the order
 * the operation was added.  This is done by:
 * 1. Visiting pixels within the image based on the order provided by the ImageTraversal iterator and
 * 2. Updating each pixel to a new color based on the ColorPicker
 * 
 * While applying the FloodFill to the image, an Animation is created by saving the image
 * after every `frameInterval` pixels are filled.  To ensure a smooth Animation, the first
 * frame is always the starting image and the final frame is always the finished image.
 * 
 * (For example, if `frameInterval` is `4` the frames are:
 *   - The initial frame
 *   - Then after the 4th pixel has been filled
 *   - Then after the 8th pixel has been filled
 *   - ...
 *   - The final frame, after all pixels have been filed)
 */
Animation FloodFilledImage::animate(unsigned frameInterval) const
{
  Animation animation;
  /** @todo [Part 2] */
  animation.addFrame(png_);
  list<ImageTraversal *>::const_iterator it_1 = traversals.begin();
  list<ColorPicker *>::const_iterator it_2 = colorpickers.begin();
  for (; it_1 != traversals.end(); it_1++, it_2++)
  {
    PNG floodpng = animation.getFrame(animation.frameCount() - 1);

    unsigned i = 0;
    for (ImageTraversal::Iterator it = (*it_1)->begin(); it != (*it_1)->end(); ++it)
    {
      HSLAPixel &current = floodpng.getPixel((*it).x, (*it).y);
      current = (*it_2)->getColor((*it).x, (*it).y);
      i++;
      if (i == frameInterval)
      {
        animation.addFrame(floodpng);
        i = 0;
      }
    }
    // if i == 0, it means the final frame is already added
    if (i != 0)
    {
      animation.addFrame(floodpng);
    }
  }
  return animation;
}
