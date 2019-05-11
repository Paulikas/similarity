#!/bin/bash
echo http://127.0.0.1:6006
tensorboard --logdir=runtime_files/logs > /dev/null 2>&1 &
