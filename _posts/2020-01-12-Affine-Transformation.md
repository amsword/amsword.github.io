---
layout: post
comments: true
title: Image Affine Transformation
---

- Affine transformation means, the point of $$(x, y)$$ in the source image is mapped to $$(x', y')$$ in the destimation image. The equation is

  $$
  \begin{bmatrix}
  x' \\ y' \\ 1
  \end{bmatrix}
  =
  M
  \begin{bmatrix}
  x \\ y \\ 1
  \end{bmatrix}
  $$

- cv2.getRotationMatrix2D(center=$$(c_x, c_y)$$, angle=$$\theta$$, scale=$$s$$)
    - the original point is not changed
    - the steps are
        - take $$(c_x, c_y)$$ as the rotation center and scale center
        - rotate the image in the clockwise direction by $$\theta$$ degrees
        - scale up the image by $$s$$.
    - explanation
        - let's first move the original point from (0, 0) to center=(cx, cy) to create
          a new coordinate system. In this system, $$(x, y) -> (x', y') = (x -
          c_x, y-c_y)$$. The matrix will be

          $$
          M_c = 
          \begin{bmatrix}
          1     &       & -c_x \\
                & 1     & -c_y \\
                &       & 1
          \end{bmatrix}
          $$
        - Then, based on the original point, let's rotate and scale the images
          by angle and s. The matrix would be

          $$
          M_{rs} = 
          \begin{bmatrix}
          s \cos(\theta)    &   s \sin(\theta)  &  \\
          -s \sin(\theta)   &  s \cos(\theta)   &  \\
                            &                   & 1 
          \end{bmatrix}
          $$

          The minus sign is in the left bottom side. To double check this, we
          can think of a point of (5, 0). If we rotate it by a small degree in
          the clockwise direction, the resulting position will be
          $$(5\cos(\theta), -5\sin(\theta))$$. Since the angle is small, the
          new $$x$$ is positive and the new $$y$$ is negative, which is
          correct.
        - After this, let's move back the original point. Thus, the matrix is
          
          $$
          M_{-c}
          \begin{bmatrix}
          1     &   &   c_x \\
                & 1 &   c_y \\
                &   &   1
          \end{bmatrix}
          $$
        - Let's combine the three matrics to get the final transformation
          matrix

          $$
          M_{-c}M_{rs}M_c = \begin{bmatrix}
          s\cos(\theta) & s\sin(\theta) & c_x(1 - s\cos(\theta)) - c_y s \sin(\theta) \\
          -s \sin(\theta) & s\cos(\theta) & c_x s \sin(\theta) + c_y (1 - s \cos(\theta)) \\
                    &       & 1
          \end{bmatrix}
          $$

          This concludes the results in the [doc](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getrotationmatrix2d)




