OBJDIR=objs
DEPDIR=deps
CXX=g++
# I don't want to have to learn csh syntax.
SHELL := /bin/bash
SRCS=\
     main.cpp \
     seqOpenmp.cpp
DEPS=$(patsubst %.cpp,$(DEPDIR)/%.d,$(SRCS))
CXXFLAGS+=-Wall -Wextra -fopenmp
LDFLAGS+=-lpthread
.PHONY: all lint clean pretty run
# all should come first in the file, so it is the default target!
all : seqOpenmp
run : seqOpenmp
        OMP_NUM_THREADS=4 ./seqOpenmp
seqOpenmp: seqOpenmp.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o seqOpenmp seqOpenmp.cpp main.cpp
clean:
	rm -rf seqOpenmp