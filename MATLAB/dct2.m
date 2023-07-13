function uhat = dct2 (u)
uhat = dct(dct(u).').'; % 2D Discrete Sine Transform
