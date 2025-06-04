######################################################### -*- Makefile -*-
# Makefile for the GiBUU Code: RootTuple library
#########################################################

.SUFFIXES:            # Delete the default suffixes


all: ;

LIBROOTTUPLE:
	install -d RootTuple-master/build
	cd RootTuple-master/build && rm -rf *
	cd RootTuple-master/build && cmake ..
	cd RootTuple-master/build && make
	@ln -f -s $(CURDIR)/RootTuple-master/build/src/libRootTuple.a $(LIBDIR)/libRootTuple.100.a

LIBROOTTUPLE_POS:
	install -d RootTuple-master/build
	cd RootTuple-master/build && rm -rf *
	cd RootTuple-master/build && cmake -DWITH_POS=TRUE ..
	cd RootTuple-master/build && make
	@ln -f -s $(CURDIR)/RootTuple-master/build/src/libRootTuple.a $(LIBDIR)/libRootTuple.100.a
