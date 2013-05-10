DEPDIR=deps
CXX=g++
# I don't want to have to learn csh syntax.
SHELL := /bin/bash

SRCS=\
     $(SRCDIR)/main.cpp \
     $(SRCDIR)/seqOpenmp.cpp

OBJS=$(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))
DEPS=$(patsubst $(SRCDIR)/%.cpp,$(DEPDIR)/%.d,$(SRCS))

CXXFLAGS+=-Wall -Wextra -fopenmp
LDFLAGS+=-lpthread

.PHONY: all lint clean pretty run

# all should come first in the file, so it is the default target!
all : seqOpenmp

run : seqOpenmp
	OMP_NUM_THREADS=4 ./seqOpenmp

seqOpenmp: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

$(OBJS): | $(OBJDIR)
$(DEPS): | $(DEPDIR)
$(OBJDIR) $(DEPDIR):
	mkdir -p $@

$(DEPDIR)/%.d: $(SRCDIR)/%.cpp Makefile
	@set -e; rm -f $@; \
        $(CXX) $(CXXFLAGS) -M $(CPPFLAGS) $< > $@.$$$$; \
        sed 's,\($*\)\.o[ :]*,$(OBJDIR)/\1.o $@ : ,g' < $@.$$$$ > $@; \
        rm -f $@.$$$$

-include $(DEPS)

clean:
	rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)
