#!/bin/bash

cd build && make -j8 && cd ../python && python3 setup.py install
python3 ../tests/python/unittest/test_auto_scheduler_feature.py

