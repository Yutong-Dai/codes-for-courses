// Your code here!
#include "Thing.h"
#include <string>
using std::string;
namespace potd
{
Thing::Thing(int size)
{
    props_max_ = size;
    props_ct_ = 0;
    properties_ = new string[size];
    values_ = new string[size];
}

void Thing::_copy(const Thing &other)
{
    props_max_ = other.props_max_;
    props_ct_ = other.props_ct_;
    properties_ = new string[other.props_max_];
    values_ = new string[other.props_max_];
    for (int i = 0; i <= props_ct_ - 1; i++)
    {
        properties_[i] = other.properties_[i];
        values_[i] = other.values_[i];
    }
}

void Thing::_destroy()
{
    if (properties_ != NULL)
    {
        delete[] properties_;
        properties_ = NULL;
    }

    if (values_ != NULL)
    {
        delete[] values_;
        values_ = NULL;
    }
}

Thing::Thing(const Thing &other)
{
    _copy(other);
}

Thing::~Thing()
{
    _destroy();
}

Thing &Thing::operator=(const Thing &other)
{
    _destroy();
    _copy(other);
    return *this;
}

int Thing::set_property(string name, string value)
{
    if (props_ct_ == 0)
    {
        properties_[0] = name;
        values_[0] = value;
        props_ct_ += 1;
        return 0;
    }
    else
    {
        for (int i = 0; i <= props_ct_ - 1; i++)
        {
            if (properties_[i] == name)
            {
                values_[i] = value;
                return i;
            }
        }
        if (props_ct_ == props_max_)
        {
            return -1;
        }
        else
        {
            properties_[props_ct_] = name;
            values_[props_ct_] = value;
            props_ct_ += 1;
            return props_ct_ - 1;
        }
    }
}
string Thing::get_property(string name)
{
    for (int j = 0; j <= props_ct_ - 1; j++)
    {
        if (name == properties_[j])
        {
            return values_[j];
        }
    }
    return "";
}

} // namespace potd
