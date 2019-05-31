#include "potd.h"

// Your code here!
double sum(vector<double>::iterator start, vector<double>::iterator end)
{
    double result = 0;
    while (start < end)
    {
        result += *start;
        start++;
    }
    return result;
}

vector<double>::iterator max_iter(vector<double>::iterator start, vector<double>::iterator end)
{
    vector<double>::iterator max_ptr = start;
    while (start < end)
    {
        if ((*start) > (*max_ptr))
        {
            max_ptr = start;
        }
        start++;
    }
    return max_ptr;
}

void sort_vector(vector<double>::iterator start, vector<double>::iterator end)
{
    double max_val;
    while (start < end)
    {
        max_val = *(max_iter(start, end));
        *(max_iter(start, end)) = *start;
        *start = max_val;
        start++;
    }
}