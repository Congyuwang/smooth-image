# Assignment 4

## Student Info
- id: 116020237
- mail: 116020237@link.cuhk.edu.cn

## Problem A
- file: [src/mask_painter.rs](src/mask_painter.rs)
- outputs:
  - [problem-a-output-1.png](test/test_outputs/problem-a-output-1.png)
  - [problem-a-output-2.png](test/test_outputs/problem-a-output-2.png)
- run script: [problem_a.sh](problem_a.sh)

### Output Images
<div style="text-align: center">
  <img src="test/test_outputs/problem-a-output-1.png" alt="output1" style="width: 200px">
  <img src="test/test_outputs/problem-a-output-2.png" alt="output1" style="width: 200px">
</div>

## Problem B
Compute the gradient:
$\nabla f = \frac{1}{2}\nabla_x\lVert{Ax-b}\rVert^2 + \frac{\mu}{2}\nabla_x\lVert{Dx}\rVert^2$,
where
$$
\begin{align*}
\nabla_x\lVert{Ax-b}\rVert^2&=\nabla_x[(Ax-b)^T(Ax-b)]\\
&=\nabla_x[x^TA^TAx-2x^TA^Tb]\\
&=2A^TAx-2A^Tb
\end{align*}
$$
and $\nabla_x\lVert{Dx}\rVert^2=\nabla_x[x^TD^TDx]=2D^TDx$.

Therefore, $\nabla f = 0$ gives $A^TAx-A^Tb+\mu D^TDx=0$,
which simplifies to $(A^TA+\mu D^TD)x=A^Tb$.
Since $\nabla f(x) = 0$ is the necessary condition for $x$ to be an optimal point.

This proves the *only if* side.

The Hessian matrix is a constant for all $x$:
$\nabla^2f=\nabla[\nabla f]=A^TA+\mu D^TD$.
Moreover, by definition, since
$x^T [\nabla^2 f] x = xA^TAx + \mu xD^TDx = \lVert{Ax}\rVert + \mu \lVert{Dx}\rVert \ge 0$,
$\nabla^2f$ is positive semi-definite everywhere.
Thus, $f$ is a convex function, and therefore $\nabla f(x) = 0$
implies $x$ to be a global minimal.

This proves the *if* side.

## Problem C

Simplify the algorithm to make it more efficient.

Let $B=A^TA+\mu D^TD$ and $c = A^T b$.
$$
\begin{align}
y^{k+1}&=x^k+{\beta_k}(x^k-x^{k-1})=(1+\beta_k)x^k-{\beta_k}x^{k-1}\\
\nabla f(y^{k+1})&=By^{k+1}-c\\
x^{k+1}&=y^{k+1}-\alpha\nabla f(y^{k+1})
\end{align}
$$
So the algorithm goes:

```
# init containers
x_old <- x.copy()
x_tmp <- zero(x.shape)
y <- zero(x.shape)

begin loop:

	# x_tmp for memorizing x
	x_tmp.copy(x)

	# x is now y^k+1
	x <- (1 + beta) * x - beta * x_old
	y.copy(x)

  # x is now Df(y^k+1)
	x <- B * x - c
	if (|x| <= tol):
  	return x_tmp
  end if
  
  # x in now x^k+1
  x <- y - alpha * x

	# put x_tmp back
	x_old <- x_tmp

end loop
```


- file: [src/ag_methods.rs](src/ag_method.rs)

## Problem D

The implementation is very optimized.
It uses as little memory allocation as possible.
Within the loop, only one memory allocation is used
for caching the matrix multiplication result of `B * p`.

- file: [src/cg_method.rs](src/cg_method.rs)

## Problem E
- files:
  - matrix generation: [src/opt_utils.rs](src/opt_utils.rs)
  - inpaint runner: [src/inpaint_worker.rs](src/inpaint_worker.rs)
- outputs:
  - CG algorithm
    - [problem-e-pens-cg.png](test/test_outputs/problem-e-pens-cg.png)
    - [problem-e-bluestreet-cg.png](test/test_outputs/problem-e-bluestreet-cg.png)
    - [problem-e-husky-cg.png](test/test_outputs/problem-e-husky-cg.png)
  - AG algorithms
    - [problem-e-pens-ag.png](test/test_outputs/problem-e-pens-ag.png)
    - [problem-e-bluestreet-ag.png](test/test_outputs/problem-e-bluestreet-ag.png)
    - [problem-e-husky-ag.png](test/test_outputs/problem-e-husky-ag.png)
- run script: [problem_e.sh](problem_e.sh)
- script output: [problem_e_output.txt](problem_e_output.txt)

### Output Images
#### CG Algorithms
<div style="text-align: center">
  <img src="test/test_outputs/problem-e-pens-cg.png" alt="problem-e-pens-cg.png" style="width: 200px">
  <img src="test/test_outputs/problem-e-bluestreet-cg.png" alt="problem-e-bluestreet-cg.png" style="width: 200px">
  <img src="test/test_outputs/problem-e-husky-cg.png" alt="problem-e-husky-cg.png" style="width: 200px">
</div>

### AG Algorithms
<div style="text-align: center">
  <img src="test/test_outputs/problem-e-pens-ag.png" alt="problem-e-pens-ag.png" style="width: 200px">
  <img src="test/test_outputs/problem-e-bluestreet-ag.png" alt="problem-e-bluestreet-ag.png" style="width: 200px">
  <img src="test/test_outputs/problem-e-husky-ag.png" alt="problem-e-husky-ag.png" style="width: 200px">
</div>
