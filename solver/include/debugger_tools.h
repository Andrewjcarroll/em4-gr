#ifndef SOLVER_DEBUGGER_TOOLS_H
#define SOLVER_DEBUGGER_TOOLS_H

#include <stdio.h>
#include <unistd.h>

static void wait_for_debugger() {
    volatile int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i) sleep(5);
}

#endif