#include <cstdio>
#include <vector>

#include "cycle_detection.cpp"

int main()
{
    // Construct graph
    Graph g(5);
    g.addEdge(0, 2);
    g.addEdge(2, 1);
    g.addEdge(3, 2);
    g.addEdge(0, 4);

    // Print the edges
    printGraph(g);

    bool cycles = hasCycles(g);
    std::cout << "Has cycles: " << cycles << std::endl;
}
