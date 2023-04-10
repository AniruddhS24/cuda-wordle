#include "util.h"

int set_base3_bit(int coloring, int pos, int value)
{
    int base = 1;
    for (int i = 0; i < pos; i++)
        base *= 3;
    coloring += value * base;
    return coloring;
}

int get_base3_bit(int coloring, int pos)
{
    for (int i = 0; i < pos; i++)
        coloring /= 3;
    return coloring % 3;
}