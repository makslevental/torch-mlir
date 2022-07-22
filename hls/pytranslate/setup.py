#!/usr/bin/python3

from distutils.core import setup, Extension

FloatToHexModule = Extension('FloatToHex',
                         sources = ['floattohexmodule.c'])

setup (name = 'FloatToHex',
       version = '1.0',
       description = 'Converts float to hex and back',
       ext_modules = [FloatToHexModule])

