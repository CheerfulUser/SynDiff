#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:35:45 2024

@author: arest
"""

import argparse
import tessvectors as tv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--test1', type=int, default=None, help='BLAAAAAA')
    parser.add_argument('--filetype', type=str, default='FFI')
    parser.add_argument('--dir', type=str, default='./')
    parser.add_argument('sector', type=int)
    parser.add_argument('camera', type=int)
    args = parser.parse_args()
    vector = tv.getvector((args.filetype, args.sector, args.camera))
    vector.to_csv('{0}TessVectors_S{1:0=3d}_C{2}_FFI.csv'.format(args.dir, args.sector, str(args.camera)))
        
    print('STEP setTESSref FINISHED')
