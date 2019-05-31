/**
 * @file Image.h
 * Contains your declaration of the interface for the Image class.
 */

#pragma once
#include "cs225/PNG.h"
using namespace cs225;
class Image : public PNG
{
  public:
    // Lighten an Image by increasing the luminance of every pixel by 0.1.
    // This function ensures that the luminance remains in the range [0, 1].
    void lighten();
    // Lighten an Image by increasing the luminance of every pixel by amount.
    void lighten(double amount);

    // Darken an Image by decreasing the luminance of every pixel by 0.1.
    // This function ensures that the luminance remains in the range [0, 1].
    void darken();
    void darken(double amount);

    // Saturates an Image by increasing the saturation of every pixel by 0.1.
    // This function ensures that the saturation remains in the range [0, 1].
    void saturate();
    void saturate(double amount);

    // Saturates an Image by decreasing the saturation of every pixel by 0.1.
    // This function ensures that the saturation remains in the range [0, 1].
    void desaturate();
    void desaturate(double amount);

    // Turns the image grayscale.
    void grayscale();

    // Rotates the color wheel by degrees.
    // Rotating in a positive direction increases the degree of the hue.
    // This function ensures that the hue remains in the range [0, 360].
    void rotateColor(double degrees);

    // Illinify the image.
    void illinify();

    // Scale the Image by a given factor.
    // For example:
    // A factor of 1.0 does not change the iamge.
    // A factor of 0.5 results in an image with half the width and half the height.
    // A factor of 2 results in an image with twice the width and twice the height.
    // This function both resizes the Image and scales the contents.
    void scale(double factor);

    // Scales the image to fit within the size (w x h).
    // This function preserves the aspect ratio of the image,
    // so the result will always be an image of width w or of height h (not necessarily both).
    // This function both resizes the Image and scales the contents.

    void scale(unsigned w, unsigned h);
};