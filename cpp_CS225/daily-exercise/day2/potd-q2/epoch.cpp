#include "epoch.h"
/* Your code here! */

int hours(time_t sec_since_epoch)
{
    int hour = sec_since_epoch / 3600;
    return hour;
}

int days(time_t sec_since_epoch)
{
    int day = sec_since_epoch / (3600 * 24);
    return day;
}

int years(time_t sec_since_epoch)
{
    int year = sec_since_epoch / (3600 * 24 * 365);
    return year;
}