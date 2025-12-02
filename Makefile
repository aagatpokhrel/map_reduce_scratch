CC = mpicxx
CFLAGS = -Wall -O3 -fopenmp

all: hybrid

hybrid: hybrid.cpp
	$(CC) $(CFLAGS) -o $@ $^ -lm
	
