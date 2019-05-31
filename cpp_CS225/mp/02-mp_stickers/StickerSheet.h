/**
 * @file StickerSheet.h
 * Contains your declaration of the interface for the StickerSheet class.
 */
#pragma once

#include "Image.h"
#include <vector>
using namespace std;

class StickerSheet
{
public:
  // Initializes this StickerSheet with a base picture and the ability to hold a
  // max number of stickers (Images) with indices 0 through max - 1.
  StickerSheet(const Image &picture, unsigned max);

  // Frees all space that was dynamically allocated by this StickerSheet.
  ~StickerSheet();

  // The copy constructor makes this StickerSheet an independent copy of the source.
  StickerSheet(const StickerSheet &other);

  // The assignment operator for the StickerSheet class.
  const StickerSheet &operator=(const StickerSheet &other);

  // Modifies the maximum number of stickers that can be stored on this StickerSheet without changing
  // existing stickers' indices. If the new maximum number of stickers is smaller than the current number
  // number of stickers, the stickers with indices above max - 1 will be lost.
  void changeMaxStickers(unsigned max);

  // Adds a sticker to the StickerSheet, so that the top-left of the sticker's Image is at (x, y) on the StickerSheet.
  // The sticker must be added to the lowest possible layer available.
  int addSticker(Image &sticker, unsigned x, unsigned y);

  // Changes the x and y coordinates of the Image in the specified layer.
  // If the layer is invalid or does not contain a sticker, this function must return false. Otherwise, this function returns true.
  bool translate(unsigned index, unsigned x, unsigned y);

  // Removes the sticker at the given zero-based layer index.
  // Make sure that the other stickers don't change order.
  void removeSticker(unsigned index);

  // Returns a pointer to the sticker at the specified index, not a copy of it.
  // That way, the user can modify the Image.
  // If the index is invalid, return NULL.
  Image *getSticker(unsigned index);

  // Renders the whole StickerSheet on one Image and returns that Image.
  // The base picture is drawn first and then each sticker is drawn in order starting with layer zero (0), then layer one (1), and so on.
  // If a pixel's alpha channel in a sticker is zero (0), no pixel is drawn for that sticker at that pixel. If the alpha channel is non-zero,
  // a pixel is drawn. (Alpha blending is awesome, but not required.)
  //  The returned Image always includes the full contents of the picture and all stickers. This means that the size of the result image may
  // be larger than the base picture if some stickers go beyond the edge of the picture.
  Image render() const;

private:
  void _copy(const StickerSheet &other);
  Image baseImage_;
  unsigned maxStickers_;
  unsigned currentStickers_;
  vector<Image *> stickers;
  vector<unsigned int> x_cord;
  vector<unsigned int> y_cord;
};
