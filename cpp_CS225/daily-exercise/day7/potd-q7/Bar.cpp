// your code here
#include <string>
#include "Bar.h"

namespace potd
{
// default constructor
Bar::Bar(std::string name)
{
    f_ = new Foo(name);
}

void Bar::_copy(const Bar &other)
{
    f_ = new Foo(*other.f_); // deep copy the concent of other (f_) and assign a heap memory to f_
}

void Bar::_destroy()
{
    if (f_ != NULL)
    {
        delete f_;
        f_ = NULL;
    }
}

Bar::Bar(const Bar &other)
{
    _copy(other);
}

Bar::~Bar()
{
    _destroy();
}

Bar &Bar::operator=(const Bar &other)
{
    if (this != &other)
    {
        _destroy(); // since assigment operator is called on two instantances, we need first to delete this.f_
        _copy(other);
    }
    return *this;
}

std::string Bar::get_name()
{
    return f_->get_name();
}
} // namespace potd
