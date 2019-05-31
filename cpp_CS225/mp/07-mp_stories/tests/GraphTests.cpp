#include "../cs225/catch/catch.hpp"

#include "../Graph.h"
#include "../Edge.h"
#include "../Vertex.h"


Graph<Vertex, Edge> createTestGraph() {
  /*
         _ b   /--------- h
        /  | _/           |
      a -- c -- e    f -- g
        \_   _/
           d 
  */

  Graph<Vertex, Edge> g;
  g.insertVertex("a");
  g.insertVertex("b");
  g.insertVertex("c");
  g.insertVertex("d");
  g.insertVertex("e");
  g.insertVertex("f");
  g.insertVertex("g");
  g.insertVertex("h");

  // Edges on `a`:
  g.insertEdge("a", "b");
  g.insertEdge("a", "c");
  g.insertEdge("a", "d");

  // Additional edges on `b`:
  g.insertEdge("b", "c");

  // Additional edges on `c`:
  g.insertEdge("c", "e");
  g.insertEdge("c", "h");

  // Additional edges on `d`:
  g.insertEdge("d", "e");

  // Additional edges on `e`: (none)

  // Additional edges on `f`:
  g.insertEdge("f", "g");

  // Additional edges on `g`:
  g.insertEdge("g", "h");

  // Additional edges on `h`: (none)

  return g;
}



TEST_CASE("Graph::numVertices() returns the vertex count", "[weight=1]") {
  Graph<Vertex, Edge> g;

  g.insertVertex("a");
  g.insertVertex("b");  
  REQUIRE( g.numVertices() == 2 );

  g.insertVertex("c");
  g.insertVertex("d");
  g.insertVertex("e");
  REQUIRE( g.numVertices() == 5 );
}

TEST_CASE("Graph::numEdges() returns the edge count", "[weight=1]") {
  Graph<Vertex, Edge> g;

  g.insertVertex("a");
  g.insertVertex("b");  
  g.insertVertex("c");
  g.insertVertex("d");
  g.insertVertex("e");

  REQUIRE( g.numEdges() == 0 );

  g.insertEdge("a", "c");
  g.insertEdge("b", "d");
  g.insertEdge("a", "e");

  REQUIRE( g.numEdges() == 3 );
}

TEST_CASE("Eight-vertex test graph has correct properties", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  REQUIRE( g.numVertices() == 8 );
  REQUIRE( g.numEdges() == 9 );
}

TEST_CASE("Graph::degree is correct", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  REQUIRE( g.degree("a") == 3 );
  REQUIRE( g.degree("c") == 4 );
  REQUIRE( g.degree("g") == 2 );
  REQUIRE( g.degree("f") == 1 );
}

TEST_CASE("Graph::incidentEdges contains incident edges (origin-only test)", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  REQUIRE( g.incidentEdges("a").size() == 3 );
}

TEST_CASE("Graph::incidentEdges contains incident edges (dest-only test)", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  REQUIRE( g.incidentEdges("h").size() == 2 );
}

TEST_CASE("Graph::incidentEdges contains incident edges (hybrid test)", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  REQUIRE( g.incidentEdges("d").size() == 2 );
}

TEST_CASE("Graph::isAdjacent is correct (same-order test)", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  REQUIRE( g.isAdjacent("a", "d") == true );
}

TEST_CASE("Graph::isAdjacent is correct (opposite-order test)", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  REQUIRE( g.isAdjacent("a", "d") == true );
}

TEST_CASE("Graph::removeEdge is correct", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  g.removeEdge("a","c");
  REQUIRE( g.numEdges() == 8 );
  REQUIRE( g.incidentEdges("a").size() == 2 );
  REQUIRE( g.incidentEdges("c").size() == 3 );
}

TEST_CASE("Graph::removeVertex is correct", "[weight=1]") {
  Graph<Vertex, Edge> g = createTestGraph();
  g.removeVertex("a");
  REQUIRE( g.numVertices() == 7 );
  REQUIRE( g.numEdges() == 6 );
  REQUIRE( g.incidentEdges("b").size() == 1 );
  REQUIRE( g.incidentEdges("c").size() == 3 );
  REQUIRE( g.incidentEdges("d").size() == 1 );
}
