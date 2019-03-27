# PyFractalExplorer

## Example

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/rY8E3OwOmJo/0.jpg)](https://www.youtube.com/watch?v=rY8E3OwOmJo)

## Controls:

W - forward

A - left

S - back

D - right

Space - up

Left CTRL - down

Q - push current speed to stack

E - restore speed from stack

Mouse - look

Mouse wheel up - increase zoom

Mouse wheel down - decrease zoom

Mouse wheel - set zoom to 1.0

left arrow - increase accuracy (lower epsilon, more details)

right arrow - decrease accuracy

up arrow - increase movement speed

down arrow - decrease movement speed

PageUp / PageDown - change fractal

Esc - exit

## Available fractals
1. Mandelbox
2. Mandelbulb
3. Sierpinski Triangle
4. Menger Sponge


## System requirements

OpenCL must be installed on your machine.

Windows: https://streamhpc.com/blog/2015-03-16/how-to-install-opencl-on-windows/

Linux: https://gist.github.com/Brainiarc7/dc80b023af5b4e0d02b33923de7ba1ed


## Installation

git clone https://github.com/kraglik/pyfractalexplorer.git

cd pyfractalexplorer

python3 -m venv ./venv

source venv/bin/activate

pip install -r requirements.txt

## Running
source venv/bin/activate

python main.py
