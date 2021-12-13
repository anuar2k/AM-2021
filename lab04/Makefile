CC     := gcc
CFLAGS := -O3 -Wall -march=native -fno-loop-interchange

.PHONY: clean

prog.out: prog.c
	$(CC) $(CFLAGS) prog.c -o prog.out

clean:
	$(RM) prog.out
