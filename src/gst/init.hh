#pragma once

#include <span>

namespace gst {

void init();

void init(std::span<char *> args);

void init(int argc, char ** argv);

}
