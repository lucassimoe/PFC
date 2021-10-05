#!/bin/sh
python main.py --pop_size 20 --pop_best 20 --sigma 0.02 --learning_rate 0.001 --decay 0.9995 --env Walker2d-v2 --it 140 && echo "Done with main.py"

python main.py --pop_size 20 --pop_best 20 --sigma 0.04 --learning_rate 0.001 --decay 0.9995 --env Walker2d-v2 --it 140 && echo "Done with main.py"

# python main.py --pop_size 20 --pop_best 20 --sigma 0.05 --learning_rate 0.001 --decay 0.9995 --env Walker2d-v2 --it 140 && echo "Done with main.py"