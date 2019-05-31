#include "cs225/PNG.h"
#include "cs225/HSLAPixel.h"
using cs225::HSLAPixel;
#include <string>
#include <cmath>

void rotate(std::string inputFile, std::string outputFile)
{
  // TODO: Part 2
  cs225::PNG input_image, output_image;
  input_image.readFromFile(inputFile);
  output_image.readFromFile(inputFile);
  int width = int(input_image.width());
  int height = int(input_image.height());
  for (int i = 0; i < width; i++)
  {
    for (int j = 0; j < height; j++)
    {
      cs225::HSLAPixel &output_pixel = output_image.getPixel(i, j);
      cs225::HSLAPixel &input_pixel = input_image.getPixel(width - 1 - i, height - 1 - j);
      output_pixel = input_pixel;
    }
  }
  output_image.writeToFile(outputFile);
}
cs225::PNG myArt(unsigned int width, unsigned int height)
{
  cs225::PNG png(width, height);
  // TODO: Part 3
  for (unsigned int i = 0; i < width; i++)
  {
    for (unsigned int j = 0; j < height; j++)
    {
      cs225::HSLAPixel &output_pixel = png.getPixel(i, j);
      // output_pixel.a = (i + j) % 360;
      // output_pixel.h = (i * j) / (width * height);
      // output_pixel.l = (i * i) / (width * height);
      // output_pixel.s = (j * j) / (width * height);
      // output_pixel.h = (i + j) % 100;
      // output_pixel.s = 0.2;
      // output_pixel.l = 0.5;
      // output_pixel.a = abs(sin(i + j));
      if (i % 100 <= 20)
      {
        output_pixel.h = 0;
        output_pixel.s = fmax(0.6, (j % 10) / 10.0);
        output_pixel.l = 0.5;
        output_pixel.a = 0.9;
      }
      else if (i % 100 <= 40)
      {
        output_pixel.h = 6;
        output_pixel.s = fmax(0.6, (j % 10) / 10.0);
        output_pixel.l = 0.5;
        output_pixel.a = 0.9;
      }
      else if (i % 100 <= 60)
      {
        output_pixel.h = 12;
        output_pixel.s = fmax(0.6, (j % 10) / 10.0);
        output_pixel.l = 0.5;
        output_pixel.a = 0.9;
      }
      else if (i % 100 <= 80)
      {
        output_pixel.h = 18;
        output_pixel.s = fmax(0.6, (j % 10) / 10.0);
        output_pixel.l = 0.5;
        output_pixel.a = 0.9;
      }
      else
      {
        output_pixel.h = 24;
        output_pixel.s = fmax(0.6, (j % 10) / 10.0);
        output_pixel.l = 0.5;
        output_pixel.a = 0.9;
      }
    }
  }
  return png;
}