#include "../cs225/catch/catch.hpp"

#include "../Graph.h"
#include "../DirectedEdge.h"
#include "../Vertex.h"

Graph<Vertex, DirectedEdge> createTestDiGraph() {
  /*
         -> b   /--------> h
        /   |  /           |
       /    v /            v
      a <-- c -> e    f -> g
        \       / 
         -> d <-
  */

  Graph<Vertex, DirectedEdge> g;
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
  g.insertEdge("a", "d");

  // Additional edges on `b`:
  g.insertEdge("b", "c");

  // Additional edges on `c`:
  g.insertEdge("c", "a");
  g.insertEdge("c", "e");
  g.insertEdge("c", "h");

  // Additional edges on `d`: (none)

  // Additional edges on `e`:
  g.insertEdge("e", "d");

  // Additional edges on `f`:
  g.insertEdge("f", "g");

  // Additional edges on `g`: (none)

  // Additional edges on `h`:
  g.insertEdge("h", "g");

  return g;
}


TEST_CASE("Graphs with `DirectedEdge`s have directed edges", "[weight=1]") {
  Graph<Vertex, DirectedEdge> g;
  g.insertVertex("a");
  g.insertVertex("b");
  g.insertEdge("a", "b");
  
  REQUIRE( g.incidentEdges("a").front().get().directed() == true );
}

TEST_CASE("Directed: eight-vertex test graph has correct properties", "[weight=1]") {
  Graph<Vertex, DirectedEdge> g = createTestDiGraph();
  REQUIRE( g.numVertices() == 8 );
  REQUIRE( g.numEdges() == 9 );
}

TEST_CASE("Directed: Graph::degree is correct", "[weight=1]") {
  Graph<Vertex, DirectedEdge> g = createTestDiGraph();
  REQUIRE( g.degree("a") == 3 );
  REQUIRE( g.degree("c") == 4 );
  REQUIRE( g.degree("g") == 2 );
  REQUIRE( g.degree("f") == 1 );
}

TEST_CASE("Directed: Graph::incidentEdges contains all incident edges", "[weight=1]") {
  Graph<Vertex, DirectedEdge> g = createTestDiGraph();
  REQUIRE( g.incidentEdges("a").size() == 3 );
  REQUIRE( g.incidentEdges("c").size() == 4 );
  REQUIRE( g.incidentEdges("d").size() == 2 );
  REQUIRE( g.incidentEdges("h").size() == 2 );
}

TEST_CASE("Directed: Graph::isAdjacent is correct (same-order test)", "[weight=1]") {
  Graph<Vertex, DirectedEdge> g = createTestDiGraph();
  REQUIRE( g.isAdjacent("c", "a") == true );
  REQUIRE( g.isAdjacent("a", "d") == true );
  REQUIRE( g.isAdjacent("f", "g") == true );
}

TEST_CASE("Directed: Graph::isAdjacent is correct (opposite-order test)", "[weight=1]") {
  Graph<Vertex, DirectedEdge> g = createTestDiGraph();
  REQUIRE( g.isAdjacent("d", "a") == false );
  REQUIRE( g.isAdjacent("g", "h") == false );
}

TEST_CASE("Directed: Graph::removeEdge is correct", "[weight=1]") {
  Graph<Vertex, DirectedEdge> g = createTestDiGraph();
  g.removeEdge("c","a");
  REQUIRE( g.numEdges() == 8 );
  REQUIRE( g.incidentEdges("a").size() == 2 );
  REQUIRE( g.incidentEdges("c").size() == 3 );
  REQUIRE( g.isAdjacent("c", "a") == false );
}

TEST_CASE("Directed: Graph::removeVertex is correct", "[weight=1]") {
  Graph<Vertex, DirectedEdge> g = createTestDiGraph();
  g.removeVertex("a");
  REQUIRE( g.numVertices() == 7 );
  REQUIRE( g.numEdges() == 6 );
  REQUIRE( g.incidentEdges("b").size() == 1 );
  REQUIRE( g.incidentEdges("c").size() == 3 );
  REQUIRE( g.incidentEdges("d").size() == 1 );
}
