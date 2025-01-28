function alt_kernel(B, m, n, alt_p, k, power)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
	if i <= m && j <= n
		B[i, j] = 0
		norm = 0
		for ki in 1:k
			d = ((alt_p[ki, 2] - i)^2 + (alt_p[ki, 1] - j)^2)^0.5
			if (d > 0)
				B[i, j] += alt_p[ki, 3] / d^power
				norm += 1 / d^power
			else
				B[i, j] = alt_p[ki, 3]
				return
			end
		end
		B[i, j] /= norm
	end
	return
end

function slope_kernel_5(A, Bx, By, m, n)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

	dfbydx::Float32 = 0.0
	dfbydy::Float32 = 0.0

	if 3 <= i <= m - 2 && 3 <= j <= n - 2
		# caluclate second order approximation of differential
		dfbydx = 1 / 3 * (4 * (A[i+1, j] - A[i-1, j]) / 2 - (A[i+2, j] - A[i-2, j]) / 4)
		dfbydy = 1 / 3 * (4 * (A[i, j+1] - A[i, j-1]) / 2 - (A[i, j+2] - A[i, j-2]) / 4)
	elseif 2 <= i <= m - 1 && 2 <= j <= n - 1
		dfbydx = (A[i+1, j] - A[i-1, j]) / 2
		dfbydy = (A[i, j+1] - A[i, j-1]) / 2
	elseif 1 <= i <= m && 1 <= j <= n
		dfbydx = 0.0
		dfbydy = 0.0
	end
	if dfbydx == 0 && dfbydy == 0
		Bx[j, i] = 0
		By[j, i] = 0
		return
	end
	Bx[j, i] = dfbydy / (dfbydx^2 + dfbydy^2)^0.5
	By[j, i] = dfbydx / (dfbydx^2 + dfbydy^2)^0.5
	return
end
