#include <iostream>
#include <string>
#include <stack>
#include <iostream>
#include "graph.cpp"

using namespace std;
// TODO: Implement this function

bool isCyclicUtil(Graph const &g, Node *node, bool visited[], bool *recStack)
{
    int v = node->value;
    if (visited[v] == false)
    {
        // Mark the current node as visited and part of recursion stack
        visited[v] = true;
        recStack[v] = true;

        // Recur for all the vertices adjacent to this vertex
        std::vector<Node *> adjNodeLits = node->outgoingNeighbors;
        std::vector<Node *>::iterator it = adjNodeLits.begin();
        for (; it != adjNodeLits.end(); it++)
        {
            cout << "checking node: " << (*it)->value << endl;
            if (!visited[(*it)->value] && isCyclicUtil(g, *it, visited, recStack))
            {
                cout << "Done with node: " << (*it)->value << endl;
                return true;
            }

            else if (recStack[(*it)->value])
            {
                cout << "Done with node: " << (*it)->value << endl;
                return true;
            }
        }
    }
    recStack[v] = false; // remove the vertex from recursion stack
    cout << "===" << endl;
    return false;
}

bool hasCycles(Graph const &g)
{

    int V = g.nodes.size();
    bool *visited = new bool[V];
    bool *recStack = new bool[V];
    for (int i = 0; i < V; i++)
    {
        visited[i] = false;
        recStack[i] = false;
    }

    for (int i = 0; i < V; i++)
    {
        Node *node = g.nodes[i];
        cout << "checking node: " << node->value << endl;
        if (isCyclicUtil(g, node, visited, recStack))
        {
            return true;
        }
    }
    return false;
}

//
//     // cout << "checking node: " << node->value << endl;
//
//     std::vector<Node *> adjNodeLits = node->outgoingNeighbors;
//     std::vector<Node *>::iterator it = adjNodeLits.begin();
//     // cout << "checking node: " << node->value << " neighbor:" << (*it)->value << endl;
//     while (it != adjNodeLits.end())
//     {
//         // int currenNeighborValue = (*it)->value;
//         // // cout << "jump to node: " << g.nodes[currenNeighbor]->value << endl;
//         // if (contains(g.nodes[currenNeighborValue]->outgoingNeighbors, *it))
//         // {
//         //     numCycles += 1;
//         // }
//         // it++;
//         searchQueue.push()
//     }
// }