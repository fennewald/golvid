#include "src/gst/init.hh"

#include <gst/gst.h>


namespace gst {

void init(void) {
	int argc = 0;
	char * argv[] = {nullptr};
	init(argc, argv);
}

void init(std::span<char *> args) {
	init(args.size(), args.data());
}

void init(int argc, char ** argv) {
	gst_init(&argc, &argv);
}

}
