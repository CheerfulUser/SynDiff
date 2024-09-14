#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:35:45 2024

@author: arest
"""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test1', type=int, default=None, help='BLAAAAAA')

    args = parser.parse_args()

    for x in range(1,100):
        print(f'HELLO {x}')
        
    print('STEP setTESSref FINISHED')