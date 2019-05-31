#include "StickerSheet.h"

StickerSheet::StickerSheet(const Image &picture, unsigned max)
{
    maxStickers_ = max;
    stickers.resize(maxStickers_);
    x_cord.resize(maxStickers_);
    y_cord.resize(maxStickers_);
    currentStickers_ = 0;
    baseImage_ = picture;
}

void StickerSheet::_copy(const StickerSheet &other)
{
    maxStickers_ = other.maxStickers_;
    currentStickers_ = other.currentStickers_;
    baseImage_ = other.baseImage_;
    stickers.assign((other.stickers).begin(), (other.stickers).end());
    x_cord.assign((other.x_cord).begin(), (other.x_cord).end());
    y_cord.assign((other.y_cord).begin(), (other.y_cord).end());
}

StickerSheet::StickerSheet(const StickerSheet &other)
{
    _copy(other);
}

const StickerSheet &StickerSheet::operator=(const StickerSheet &other)
{
    if (this != &other)
    {
        _copy(other);
    }
    return *this;
}

void StickerSheet::changeMaxStickers(unsigned max)
{
    stickers.resize(max);
    x_cord.resize(max);
    y_cord.resize(max);

    if (max - 1 < currentStickers_)
    {
        currentStickers_ = max;
    }
    maxStickers_ = max;
}

int StickerSheet::addSticker(Image &sticker, unsigned x, unsigned y)
{
    if (currentStickers_ < maxStickers_)
    {
        x_cord.insert(x_cord.begin() + currentStickers_, x);
        y_cord.insert(y_cord.begin() + currentStickers_, y);
        stickers.insert(stickers.begin() + currentStickers_, &sticker);
        currentStickers_++;
        return currentStickers_ - 1;
    }
    else
    {
        return -1;
    }
}

bool StickerSheet::translate(unsigned index, unsigned x, unsigned y)
{
    if (index >= currentStickers_)
    {
        return false;
    }
    else
    {
        x_cord[index] = x;
        y_cord[index] = y;
        return true;
    }
}

void StickerSheet::removeSticker(unsigned index)
{
    if (index < maxStickers_)
    {
        stickers.erase(stickers.begin() + index);
        x_cord.erase(x_cord.begin() + index);
        y_cord.erase(y_cord.begin() + index);
        currentStickers_--;
    }
}

Image *StickerSheet::getSticker(unsigned index)
{
    if (index >= maxStickers_)
    {
        return NULL;
    }
    else
    {
        return (stickers[index]);
    }
}

Image StickerSheet::render() const
{
    Image result = baseImage_;
    unsigned w = result.width();
    unsigned h = result.height();

    // Check if any sticker goes beyond the edge of the base picture
    for (unsigned i = 0; i < currentStickers_; i++)
    {
        if (x_cord[i] + stickers[i]->width() > w)
        {
            w = x_cord[i] + stickers[i]->width();
        }
        if (y_cord[i] + stickers[i]->height() > h)
        {
            h = y_cord[i] + stickers[i]->height();
        }
    }

    // Scale the image
    if (w > result.width() || h > result.height())
    {
        double factor = std::max((w + 1) * 1.0 / result.width(), (h + 1) * 1.0 / result.height());
        result.scale(factor);
    }

    // Draw the stickers
    for (unsigned i = 0; i < currentStickers_; i++)
    {
        for (unsigned x = 0; x < stickers[i]->width(); x++)
        {
            for (unsigned y = 0; y < stickers[i]->height(); y++)
            {
                HSLAPixel &sticker_pixel = stickers[i]->getPixel(x, y);
                if (sticker_pixel.a > 0)
                {
                    HSLAPixel &result_pixel = result.getPixel(x_cord[i] + x, y_cord[i] + y);
                    result_pixel = sticker_pixel;
                }
            }
        }
    }

    return result;
}

StickerSheet::~StickerSheet()
{
    vector<unsigned>().swap(x_cord);
    vector<unsigned>().swap(y_cord);
    vector<Image *>().swap(stickers);
}