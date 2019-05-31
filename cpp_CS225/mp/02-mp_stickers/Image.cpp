#include "Image.h"
#include <iostream>
#include <cmath>

void Image::lighten()
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            pixel.l = fmin(1.0, pixel.l + 0.1);
        }
    }
}

void Image::lighten(double amount)
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            pixel.l = fmin(1.0, pixel.l + amount);
        }
    }
}

void Image::darken()
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            double temp = pixel.l - 0.1;
            pixel.l = fmax(0.0, pixel.l - 0.1);
        }
    }
}

void Image::darken(double amount)
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            double temp = pixel.l - amount;
            pixel.l = fmax(0.0, pixel.l - amount);
        }
    }
}

void Image::saturate()
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            pixel.s = fmin(1.0, pixel.s + 0.1);
        }
    }
}

void Image::saturate(double amount)
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            pixel.s = fmin(1.0, pixel.s + amount);
        }
    }
}

void Image::desaturate()
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            pixel.s = fmax(0.0, pixel.s - 0.1);
        }
    }
}

void Image::desaturate(double amount)
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            pixel.s = fmax(0.0, pixel.s - amount);
        }
    }
}

void Image::grayscale()
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            pixel.s = 0;
        }
    }
}

void Image::rotateColor(double degrees)
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            double temp = pixel.h + degrees;
            if (temp >= 0 and temp <= 360.0)
            {
                pixel.h += degrees;
            }
            else if (temp > 360)
            {
                pixel.h = temp - 360;
            }
            else
            {
                pixel.h = temp + 360;
            }
        }
    }
}

void Image::illinify()
{
    for (unsigned x = 0; x < this->width(); x++)
    {
        for (unsigned y = 0; y < this->height(); y++)
        {
            HSLAPixel &pixel = this->getPixel(x, y);
            if (pixel.h > 113.5 and pixel.h < 293.5)
            {
                pixel.h = 216.0;
            }
            else
            {
                pixel.h = 11.0;
            }
        }
    }
}

void Image::scale(double factor)
{
    // reference: p54-p56 of <https://ia802707.us.archive.org/23/items/Lectures_on_Image_Processing/EECE_4353_15_Resampling.pdf>
    // https://stackoverflow.com/questions/32124170/bilinear-image-interpolation-scaling-a-calculation-example

    unsigned int out_width = int(this->width() * factor);
    unsigned int out_height = int(this->height() * factor);
    double ratio = 1.0 / factor;
    PNG *out_image = new PNG(out_width, out_height);

    for (unsigned int c = 0; c < out_width; c++)
    {
        double cf = c * ratio;
        int c0 = int(cf);
        double delta_c = cf - c0;
        int c1 = fmin(c0 + 1, this->width() - 1);

        for (unsigned int r = 0; r < out_height; r++)
        {
            // bi-linear interpolation
            double rf = r * ratio;
            int r0 = int(rf);
            double delta_r = rf - r0;
            int r1 = fmin(r0 + 1, this->height() - 1);

            HSLAPixel &p_00 = this->getPixel(c0, r0);
            HSLAPixel &p_01 = this->getPixel(c0, r1);
            HSLAPixel &p_10 = this->getPixel(c1, r0);
            HSLAPixel &p_11 = this->getPixel(c1, r1);
            HSLAPixel &out_pixle = out_image->getPixel(c, r);

            out_pixle.h = p_00.h * (1 - delta_r) * (1 - delta_c) + p_10.h * delta_r * (1 - delta_c) + p_01.h * (1 - delta_r) * delta_c + p_11.h * delta_r * delta_c;
            out_pixle.s = p_00.s * (1 - delta_r) * (1 - delta_c) + p_10.s * delta_r * (1 - delta_c) + p_01.s * (1 - delta_r) * delta_c + p_11.s * delta_r * delta_c;
            out_pixle.l = p_00.l * (1 - delta_r) * (1 - delta_c) + p_10.l * delta_r * (1 - delta_c) + p_01.l * (1 - delta_r) * delta_c + p_11.l * delta_r * delta_c;
            out_pixle.a = p_00.a * (1 - delta_r) * (1 - delta_c) + p_10.a * delta_r * (1 - delta_c) + p_01.a * (1 - delta_r) * delta_c + p_11.a * delta_r * delta_c;

            out_pixle.h = out_pixle.h > 360 ? 360 : out_pixle.h;
            out_pixle.s = out_pixle.s > 1.0 ? 1.0 : out_pixle.s;
            out_pixle.l = out_pixle.l > 1.0 ? 1.0 : out_pixle.l;
            out_pixle.a = out_pixle.a > 1.0 ? 1.0 : out_pixle.a;
        }
    }

    this->resize(out_width, out_height);
    for (unsigned r = 0; r < this->height(); r++)
    {
        for (unsigned c = 0; c < this->width(); c++)
        {
            this->getPixel(c, r) = out_image->getPixel(c, r);
        }
    }

    delete out_image;
}

void Image::scale(unsigned int w, unsigned int h)
{
    double factor;
    if (w * (this->height()) / (this->width()) == h)
    {
        factor = w / this->width();
    }
    else if (w * (this->height()) / (this->width()) > h)
    {
        factor = h / this->height();
    }
    else
    {
        factor = w / this->width();
    }

    unsigned int out_width = int(this->width() * factor);
    unsigned int out_height = int(this->height() * factor);
    double ratio = 1.0 / factor;
    PNG *out_image = new PNG(out_width, out_height);
    for (unsigned int r = 0; r < out_height; r++)
    {
        double rf = r * ratio;
        int r0 = int(rf);
        double delta_r = rf - r0;
        int r1 = fmin(r0 + 1, this->height() - 1);
        for (unsigned int c = 0; c < out_width; c++)
        {
            // bi-linear interpolation

            double cf = c * ratio;
            int c0 = int(cf);
            double delta_c = cf - c0;
            int c1 = fmin(c0 + 1, this->width() - 1);

            HSLAPixel &p_00 = this->getPixel(r0, c0);
            HSLAPixel &p_01 = this->getPixel(r0, c1);
            HSLAPixel &p_10 = this->getPixel(r1, c0);
            HSLAPixel &p_11 = this->getPixel(r1, c1);
            HSLAPixel &out_pixle = out_image->getPixel(r, c);

            out_pixle.h = p_00.h * (1 - delta_r) * (1 - delta_c) + p_10.h * delta_r * (1 - delta_c) + p_01.h * (1 - delta_r) * delta_c + p_11.h * delta_r * delta_c;
            out_pixle.s = p_00.s * (1 - delta_r) * (1 - delta_c) + p_10.s * delta_r * (1 - delta_c) + p_01.s * (1 - delta_r) * delta_c + p_11.s * delta_r * delta_c;
            out_pixle.l = p_00.l * (1 - delta_r) * (1 - delta_c) + p_10.l * delta_r * (1 - delta_c) + p_01.l * (1 - delta_r) * delta_c + p_11.l * delta_r * delta_c;
            out_pixle.a = p_00.a * (1 - delta_r) * (1 - delta_c) + p_10.a * delta_r * (1 - delta_c) + p_01.a * (1 - delta_r) * delta_c + p_11.a * delta_r * delta_c;

            out_pixle.h = out_pixle.h > 360 ? 360 : out_pixle.h;
            out_pixle.s = out_pixle.s > 1.0 ? 1.0 : out_pixle.s;
            out_pixle.l = out_pixle.l > 1.0 ? 1.0 : out_pixle.l;
            out_pixle.a = out_pixle.a > 1.0 ? 1.0 : out_pixle.a;
        }
    }

    this->resize(out_width, out_height);
    for (unsigned r = 0; r < this->height(); r++)
    {
        for (unsigned c = 0; c < this->width(); c++)
        {
            this->getPixel(r, c) = out_image->getPixel(r, c);
        }
    }

    delete out_image;
}