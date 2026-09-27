# Ultra-fast 3D-MHD Equilibrium solver
This ```repo``` is the result of my master internship-thesis at INRIA (FR) where I implemented from scratch [this](https://iopscience.iop.org/article/10.1088/1741-4326/ae2937) paper 
and I made the extension to full **continous parametric** for different kinds of fusion reactors, both in pressure amplitude $p(\rho)$ and rotational transoform $\iota(\rho)$

Having a full continuous parametric solver means that once the neural network has been trained, it's possible to have, for a given parameter space $\mathcal{P}$, an almost infinite number 
of solutions at an inference cost of a few milliseconds, enabling real-time fusion plasma control 
