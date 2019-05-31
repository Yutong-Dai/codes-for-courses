#include "Image.h"
#include "StickerSheet.h"

int main()
{

  //
  // Reminder:
  //   Before exiting main, save your creation to disk as myImage.png
  //
  Image png, sticker_tree, sticker_c, sticker_boom, result;
  png.readFromFile("alma.png");
  sticker_tree.readFromFile("tree.png");
  sticker_boom.readFromFile("boom.png");
  sticker_c.readFromFile("c.png");

  png.scale(2.0);
  sticker_tree.scale(0.3);
  sticker_boom.scale(0.3);
  sticker_c.scale(0.1);

  StickerSheet alma(png, 3);
  alma.addSticker(sticker_tree, 580, 45);
  alma.addSticker(sticker_c, 820, 30);
  alma.addSticker(sticker_boom, 1050, 40);

  result = alma.render();

  result.writeToFile("myImage.png");
  return 0;
}
