/**
 * @file maptiles.cpp
 * Code for the maptiles function.
 */

#include <iostream>
#include <map>
#include "maptiles.h"
//#include "cs225/RGB_HSL.h"

using namespace std;

Point<3> convertToXYZ(LUVAPixel pixel)
{
    return Point<3>(pixel.l, pixel.u, pixel.v);
}

MosaicCanvas *mapTiles(SourceImage const &theSource,
                       vector<TileImage> &theTiles)
{
    /**
     * @todo Implement this function!
     */
    int rows = theSource.getRows();
    int cols = theSource.getColumns();
    MosaicCanvas *mosaic = new MosaicCanvas(rows, cols);
    // Use the tiles to build a kd tree
    // Meanwhile use a std::map to map average colors to tiles.
    vector<Point<3>> tilesColor;
    map<Point<3>, TileImage *> tilesMap;
    for (TileImage &tile : theTiles)
    {
        Point<3> tilecolor = convertToXYZ(tile.getAverageColor());
        tilesColor.push_back(tilecolor);
        tilesMap[tilecolor] = &tile;
    }
    KDTree<3> tileTree(tilesColor);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            Point<3> query = convertToXYZ(theSource.getRegionColor(i, j));
            TileImage *curBest = tilesMap[tileTree.findNearestNeighbor(query)];
            mosaic->setTile(i, j, curBest);
        }
    }

    return mosaic;
}
