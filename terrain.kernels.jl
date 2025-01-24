function alt_kernel(B, m, n, alt_p, k, power)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if i <= m && j <= n
        B[i, j] = 0
        norm = 0
        for ki in 1:k
            d = ((alt_p[ki, 2] - i)^2 + (alt_p[ki, 1] - j)^2)^0.5
            if (d > 0)
                B[i,j] += alt_p[ki, 3]/d^power
                norm += 1/d^power
            else
                B[i,j] = alt_p[ki, 3]
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
		if 3 <= i <= m-2 && 3 <= j <= n-2
			# caluclate second order approximation of differential
			xph = A[i+1,j]
			xmh = A[i-1,j]
			xp2h = A[i+2,j]
			xm2h = A[i-2,j]

			yph = A[i,j+1]
			ymh = A[i,j-1]
			yp2h = A[i,j+2]
			ym2h = A[i,j-2]

			dfbydx = 1/3*(4*(xph-xmh)/2 - (xp2h-xm2h)/4)
			dfbydy = 1/3*(4*(yph-ymh)/2 - (yp2h-ym2h)/4)

			# B[i, j] = atan(dfbydy, dfbydx)
			norm = (dfbydx^2+dfbydy^2)^0.5
			Bx[j, i] = dfbydy/norm
			By[j, i] = dfbydx/norm
		elseif 2 <= i <= m-1 && 2 <= j <= n-1
			xph = A[i+1,j]
			xmh = A[i-1,j]

			yph = A[i,j+1]
			ymh = A[i,j-1]

			dfbydx = (xph-xmh)/2
			dfbydy = (yph-ymh)/2

			# B[j, i] = atan(dfbydy, dfbydx)
			norm = (dfbydx^2+dfbydy^2)^0.5
			Bx[j, i] = dfbydy/norm
			By[j, i] = dfbydx/norm
		elseif 1 <= i <= m && 1 <= j <= n
			Bx[j, i] = 0.0
			By[j, i] = 0.0
		end
		return
	end