Here you find a version of the RootTuple library, as it may be
downloaded from https://roottuple.hepforge.org/
but adopted to the needs of GiBUU!

-------------------------------------------------------------------------
If you got this file via the GiBUU installation instructions,
you do not have to do anything here (except when your your compiler is
complaining about something), since the usual GiBUU installation
allows you to get all of it run via
{{{
make buildRootTuple
}}}
-------------------------------------------------------------------------

If the above does not work for you, you may try to install it by hand:

In order to get it running with GiBUU and your ROOT installation, it
is necessary to change one line in `RootTuple-master/src/CMakeLists.txt`:
{{{
#add_library (RootTuple SHARED ${RootTuple_SOURCES})
add_library (RootTuple STATIC ${RootTuple_SOURCES})
}}}

After this, you just 'build' the static library and copy it to a place
where GiBUU can find it, instead of 'installing' the library:

{{{
mkdir build; cd build
cmake ..
make
cp src/libRootTuple.a <your/path/to/GiBUU>/LIB/lib/libRootTuple.100.a
}}}

where `<your/path/to/GiBUU>` may be something like `~/GiBUU/release/`.
