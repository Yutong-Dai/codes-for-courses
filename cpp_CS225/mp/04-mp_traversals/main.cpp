
#include "cs225/PNG.h"
#include "FloodFilledImage.h"
#include "Animation.h"

#include "imageTraversal/DFS.h"
#include "imageTraversal/BFS.h"

#include "colorPicker/RainbowColorPicker.h"
#include "colorPicker/GradientColorPicker.h"
#include "colorPicker/GridColorPicker.h"
#include "colorPicker/SolidColorPicker.h"
#include "colorPicker/MyColorPicker.h"

using namespace cs225;

int main()
{

  // @todo [Part 3]
  // - The code below assumes you have an Animation called `animation`
  // - The code provided below produces the `myFloodFill.png` file you must
  //   submit Part 3 of this assignment -- uncomment it when you're ready.

  PNG myfig;
  myfig.readFromFile("./test.png");

  PNG png(myfig.width(), myfig.height());
  BFS bfs(png, Point(0, 0), 1);

  // DFS dfs(myfig, Point(10, 10), 0.4); //  Point(100, 90)
  // DFS dfs2(myfig, Point(myfig.width() - 1, myfig.height() - 1), 0.4);
  DFS dfs3(myfig, Point(0, myfig.height() - 1), 0.17);

  MyColorPicker myfigcolor(myfig);
  SolidColorPicker solid(HSLAPixel(120, 1, 0.9));
  RainbowColorPicker rainbow(0.2);

  FloodFilledImage myfigflood(png);
  myfigflood.addFloodFill(bfs, myfigcolor);
  // myfigflood.addFloodFill(dfs, solid);
  // myfigflood.addFloodFill(dfs2, rainbow);
  myfigflood.addFloodFill(dfs3, rainbow);
  // myfigflood.addFloodFill(bfs2, rainbow);

  // myfigflood.addFloodFill(bfs2, grid);
  Animation animation = myfigflood.animate(1500);

  PNG lastFrame = animation.getFrame(animation.frameCount() - 1);
  lastFrame.writeToFile("mytest.png");
  animation.write("mytest.gif");
  return 0;
}
