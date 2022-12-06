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
![output1](test/test_outputs/problem-a-output-1.png)
![output2](test/test_outputs/problem-a-output-2.png)

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

This proves the $if$ side.

## Problem C

Simplify the algorithm to make it more efficient.

Let $B=A^TA+\mu D^TD$ and $c = A^T b$. Combine $y^{k+1}=x^k+(1-\beta_k) x^{k-1}$ and
$x^{k+1}=y^{k+1}-\alpha(By^{k+1}-c)=(I-\alpha B)y^{k+1}+\alpha c$,
we get: $x^{k+1}=(I-\alpha B)[(1 + \beta_k)x^k - \beta_k x^{k-1}]+\alpha c$.

That is $x^{k+1}=(1 + \beta_k)(I-\alpha B)x^k - \beta_k(I-\alpha B)x^{k-1}+\alpha c$.

Let $Z=I-\alpha B$ and $U=\alpha c$.

We have $x^{k+1}=(1+\beta_k)Zx^k-\beta_kZx^{k-1}+U$.

- file: [src/ag_methods.rs](src/ag_method.rs)

## Problem D

The implementation is very optimized.
It uses as little memory allocation as possible.
Within the loop, only one memory allocation is used
for caching the matrix multiplication result of `B * p`.

- file: [src/cg_method.rs](src/cg_method.rs)

## Problem E

Solve
