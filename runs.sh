#!/bin/sh
python main.py --seed 42 --pop_size 40 --pop_best 40 --sigma 0.01 --learning_rate 0.001 --decay 0.995 --env Ant-v2 --it 50 && echo "Done with main.py 1"
python main.py --seed 42 --pop_size 40 --pop_best 30 --sigma 0.01 --learning_rate 0.001 --decay 0.995 --env Ant-v2 --it 50 && echo "Done with main.py 3"
python main.py --seed 42 --pop_size 40 --pop_best 20 --sigma 0.01 --learning_rate 0.001 --decay 0.995 --env Ant-v2 --it 50 && echo "Done with main.py 4"
python main.py --seed 42 --pop_size 40 --pop_best 10 --sigma 0.01 --learning_rate 0.001 --decay 0.995 --env Ant-v2 --it 50 && echo "Done with main.py 5"

python main.py --seed 42 --pop_size 20 --pop_best 20 --sigma 0.02 --learning_rate 0.001 --decay 0.9995 --env Swimmer-v2 --it 50 && echo "Done with main.py 1"
python main.py --seed 42 --pop_size 20 --pop_best 15 --sigma 0.02 --learning_rate 0.001 --decay 0.9995 --env Swimmer-v2 --it 50 && echo "Done with main.py 3"
python main.py --seed 42 --pop_size 20 --pop_best 10 --sigma 0.02 --learning_rate 0.001 --decay 0.9995 --env Swimmer-v2 --it 50 && echo "Done with main.py 4"
python main.py --seed 42 --pop_size 20 --pop_best 5 --sigma 0.02 --learning_rate 0.001 --decay 0.9995 --env Swimmer-v2 --it 50 && echo "Done with main.py 5"
