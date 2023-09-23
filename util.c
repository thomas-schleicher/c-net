//
// Created by jakob on 21.09.2023.
//

#include "util.h"
#include <stdio.h>
void updateBar(int percent_done){
    const int BAR_LENGTH = 30;
    int numChar = percent_done * BAR_LENGTH / 100;

    printf("\r[");
    for(int i = 0; i < numChar; i++){
        printf("#");
    }
    for(int i = 0; i < BAR_LENGTH - numChar; i++){
        printf(".");
    }
    printf("] (%d%%)", percent_done);
    fflush(stdout);
}
