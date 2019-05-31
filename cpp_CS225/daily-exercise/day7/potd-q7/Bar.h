// your code here
#pragma once

#include <string>
#include "Foo.h"

namespace potd
{
class Bar
{
  public:
    Bar(std::string name);
    Bar(const Bar &other); // copy constructor
    Bar &operator=(const Bar &other);
    ~Bar();
    std::string get_name();

  private:
    Foo *f_;
    void _copy(const Bar &other);
    void _destroy();
};
} // namespace potd
