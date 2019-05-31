#include <iostream>
#include <string>
#include "adjacency_list.h"

using namespace std;

bool containsEdge(Graph const *const g, int src, int dest)
{
        // Your code here
        // cout << "Move to vertex: " << g->adjLists[src].vertex << endl;
        LinkedListNode *current = g->adjLists[src].edges;
        // cout << "current node: " << g->adjLists[src].edges->v << endl;
        if (current == NULL)
        {
                return false;
        }
        while (current != NULL)
        {
                if (current->v == dest)
                {
                        return true;
                }
                else
                {
                        current = current->next;
                }
        }
        return false;
}

void addEdge(Graph *g, int src, int dest)
{
        // Your code here
        if (!containsEdge(g, src, dest))
        {
                LinkedListNode *newHead = new LinkedListNode;
                newHead->v = dest;
                newHead->next = g->adjLists[src].edges;
                g->adjLists[src].edges = newHead;
        }

        // cout << "add: " << g->adjLists[src].edges->v << endl;
}

int numOutgoingEdges(Graph const *const g, int v)
{
        // Your code here
        int numOut = 0;
        LinkedListNode *current = g->adjLists[v].edges;
        if (current == NULL)
        {
                return numOut;
        }
        while (current != NULL)
        {

                current = current->next;
                numOut += 1;
        }
        return numOut;
}

int numIncomingEdges(Graph const *const g, int v)
{
        // Your code here
        int numOut = 0;
        for (int i = 0; i < g->n; i++)
        {
                if (containsEdge(g, g->adjLists[i].vertex, v))
                {
                        numOut += 1;
                }
        }
        return numOut;
}

// No need to modify the functions below

Graph *createVertices(int numV)
{
        Graph *g = new Graph();
        g->adjLists = new AdjacencyList[numV];
        g->n = numV;

        // Initialize the vertices
        for (int i = 0; i < numV; i += 1)
        {
                g->adjLists[i].vertex = i;
                g->adjLists[i].edges = NULL;
        }

        return g;
}

void printGraph(Graph const *const g)
{
        for (int i = 0; i < g->n; i += 1)
        {
                AdjacencyList adjList = g->adjLists[i];

                int v = adjList.vertex;
                // Vertex
                cout << "Vertex: " << v << endl;

                // Print number of incoming and outgoing edges
                int inc = numIncomingEdges(g, v);
                int out = numOutgoingEdges(g, v);
                cout << "Number of incoming edges: " << inc << endl;
                cout << "Number of outgoing edges: " << out << endl;

                cout << "\n"
                     << endl;
        }
}
