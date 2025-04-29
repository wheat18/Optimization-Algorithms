def conjugate_gradient(x0, Q, b, tol=1e-5, n=10000):
    x = x0.copy() #
    r =  Q @ x - b
    d = -r
    k = 0
    while np.linalg.norm(r) >= tol and k < n:
      Qd = Q@d
      alpha = np.dot(r.T,r) / np.dot(d.T,Qd)
      x_next = x + alpha * d

      r_next = r + alpha * Qd

      beta_next = np.dot(r_next.T, r_next) / np.dot(r.T, r)

      d_next = -r_next + beta_next * d

      # Updating the values for the next iteration

      x = x_next
      r = r_next
      d = d_next

      k += 1

    return x
