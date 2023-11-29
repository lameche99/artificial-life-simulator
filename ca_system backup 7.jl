### A Pluto.jl notebook ###
# v0.19.30

#> [[frontmatter.author]]
#> name = "Kishore Shenoy"
#> url = "kichappa.github.io"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 4727903f-a54b-4d73-8998-fa99bb2481aa
md"# CA for Topography and Enemies"

# ╔═╡ c8c9a170-7cc7-4bb3-b9dc-1654f4c2cefd
begin
	# code to display a 2D array as an image
	function show_image(A, color_range=:viridis)
		heatmap(1:size(A, 1), 1:size(A, 2), A, aspect_ratio=:equal, color=color_range, backend=:gr)
	end
	md"Defining show_image() that can plot the 2D version of our model." 
end

# ╔═╡ df27f8a4-f258-43b4-acdc-b8ea0f9ffc88
md"## Initial State"

# ╔═╡ e5c741d7-7c52-4097-8d02-89d76495d53f
function neighbour_sum(A, pos)
	i, j=pos
	neighbours = [[i-1,j],[i-1,j+1],[i,j+1],[i+1,j+1],[i+1,j],[i+1,j-1],[i,j-1],[i-1,j-1]]
	sum=0
	for neighbour in neighbours
		i, j = neighbour
		if(i>0 && i<=size(A, 1) && j>0 && j<=size(A, 2))
			# println("A($i, $j)=")
			# println("$(A[i, j])\n")
			sum+=A[i,j]
		end
	end
	return sum
end

# ╔═╡ 29fb1a62-86bf-4bab-bb7e-dbbfd5024917
function conway(A)
	m, n = size(A)
	B=copy(A)
	# for t in 1:T
		for i in 1:m
			for j in 1:n
				# life_decision = 0
				# for di in -1:1, dj in -1:1
				# 	i_n, j_n = i + di, j + dj
				# 	if 1 <= i_n <= m && 1 <= j_n <= n && !(di==0 && dj===0)
				# 		life_decision += A[i_n, j_n]
				# 	end
				# end
				
				# if life_decision < 2 || life_decision > 3
				# 	B[i, j] = 0
				# elseif life_decision == 3
				# 	B[i, j] = 1
				# else
				# 	B[i, j] = A[i, j]
				# end
				life_decision = neighbour_sum(A, [i,j])
				if(life_decision < 2 || life_decision > 3)
					B[i,j] = 0
				elseif(life_decision === 3)
					B[i,j]=1
				else
					B[i,j]=B[i,j]
				end
			end
		end
	# end
	return B
end	

# ╔═╡ fd3512a7-8d52-4d25-9ad8-0cc80555da7f
function kernel(A, B, m, n)
	i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
	j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
	if i <= m && j <= n
		# B[i,j]=A[i, j]
		life_decision = 0
		for di in -1:1, dj in -1:1
			i_n, j_n = i + di, j + dj
			if 1 <= i_n <= m && 1 <= j_n <= n && !(di==0 && dj===0)
				life_decision += A[i_n, j_n]
			end
		end
		
		if life_decision < 2 || life_decision > 3
			B[i, j] = 0
		elseif life_decision == 3
			B[i, j] = 1
		else
			B[i, j] = A[i, j]
		end
	end
	return
end

# ╔═╡ 2a3753d3-c08c-4e85-907e-9ebb5a67dab3
function conway_gpu(A)
    m, n = size(A)
	A = CuArray(A)
    B = similar(A)  # Create a GPU array of the same size and type as A

	threads_x = min(32, m)  # Limit to 32 threads in the x dimension
    threads_y = min(32, n)  # Limit to 32 threads in the y dimension
    blocks_x = ceil(Int, m / threads_x)
    blocks_y = ceil(Int, n / threads_y)
	
 	@cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) kernel(A, B, m, n)
    
    return collect(B)
end

# ╔═╡ 2fe91b37-1c3f-49ce-bfa2-702a180b78a0
begin
	md"``X``, ``Y`` subdivisions, ``n`` = $(@bind n NumberField(0:100; default=100))"
end

# ╔═╡ 8327cfec-51df-4c38-839a-b7212ddb24e7
md"``X_{\max}, Y_{\max}``, L = $(@bind L NumberField(0:100; default=100))"

# ╔═╡ 701891a4-6a87-427e-af9b-487dec1dee4d
md"Time of simulation, ``T_{\text{max}}``"

# ╔═╡ 0f344406-4816-4cd6-ae8e-83a8b918fa11
function next_pos(current_pos, B, seed)
	i, j=current_pos

	# # println("Seed = $seed")
	# # Random.seed!(seed)
	# d = rand(-1:1, (2,1))
	
	# println("d=$d")
	# println("x0, y0= $(current_pos)")
	# x, y=current_pos+d
	# println("Old x, y= $([x, y])")

	neighbors = [[-1,0],[0-1,0+1],[0,0+1],[0+1,0+1],[0+1,0],[0+1,0-1],[0,0-1],[0-1,0-1]]
	direction = [0, 0]
	for neighbor in neighbors
		i_n, j_n = neighbor
		direction += B[i_n+i, j_n+j]*[i_n, j_n]
	end
	# println("Direction=$direction")
	direction[1] = sign(direction[1])*ceil(abs(direction[1])/8)
	direction[2] = sign(direction[2])*ceil(abs(direction[2])/8)
	direction = 1 * direction
	# println("NormDirection=$direction")

	if direction[1]==0 && direction[2]==0
		direction = rand(-1:1, (2,1))
	end
	
	x = i + direction[1]
	y = j + direction[2]
	
	x=min(max(x, 1), n)
	y=min(max(y, 1), n)
	# println("New x, y= $([x, y])")
	return [x, y]
end

# ╔═╡ 4ec0a200-78df-4cfd-9efe-105dad6f4ef3
function encode_agent(agent_pos, B)
	B_out = copy(B)
	x, y = agent_pos
	B_out[x, y] = 2
	return B_out
end

# ╔═╡ fffa26a7-ecf6-4be0-ab7c-423665caf7a5
md"## Topography"

# ╔═╡ 72a7cb99-5483-4c82-9554-007c2ba44413
md"Number of height points, $(@bind altPs NumberField(0:100; default=7))"

# ╔═╡ cd4ee775-74d9-417f-9c97-6c8d321d7580
md"Max height, $(@bind max_height NumberField(0:100; default=L/10))"

# ╔═╡ ba6660df-59b7-4c70-b30f-b8548d63b0d2
begin
	function alt_kernel(A, B, m, n, alt_p, k, power)
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
	
	function topography_gpu(A, alt_p, power)
	    m, n = size(A)
		k, _ = size(alt_p)
		A_gpu = CuArray(A)
	    B = similar(A_gpu)  # Create a GPU array of the same size and type as A
	
		threads_x = min(32, m)  # Limit to 32 threads in the x dimension
	    threads_y = min(32, n)  # Limit to 32 threads in the y dimension
	    blocks_x = ceil(Int, m / threads_x)
	    blocks_y = ceil(Int, n / threads_y)
		
	 	@cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) alt_kernel(A_gpu, B, m, n, CuArray(alt_p), k, power)
	    
	    return collect(B)
	end
	md"Kernel and Method to generate Topography"
end

# ╔═╡ 82d0e800-deb1-42fe-b1d3-2018d8639ff8
md"neighbourhood radius, `n_radius` $(@bind n_radius NumberField(0:1000; default=3))"

# ╔═╡ 8f0937f0-813b-4256-a8b9-afb22e092a42
md"Topography of the system"

# ╔═╡ 6d4076dc-68c8-42f8-a43e-222e3410bdbf
md"Topography contour"

# ╔═╡ 1add5389-3a8b-40b7-b999-8df22bb45900
begin
	function plot_topo_kernel(topo, bushes, out, m, n)
		i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
		j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
		if 1 <= i <= m && 1 <= j <= n
			out[i, j] = topo[i,j] + bushes[i,j]*1
		end
		return
	end
	
	function plot_topo_gpu(topo, bushes)
	    m, n = size(topo)
		topo_gpu = CuArray(topo)
		bushes_gpu = CuArray(bushes)
		output = similar(topo_gpu)
		threads_x = min(32, m)  # Limit to 32 threads in the x dimension
	    threads_y = min(32, n)  # Limit to 32 threads in the y dimension
	    blocks_x = ceil(Int, m / threads_x)
	    blocks_y = ceil(Int, n / threads_y)
		
	 	@cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) plot_topo_kernel(topo_gpu, bushes_gpu, output, m, n)
	    
	    return collect(output)
	end
	md"Kernel and GPU handler for superposing bushes onto the topography"
end

# ╔═╡ 11f7bf70-4a39-451c-9bdb-9369742dcce0
md"Random Seed, $(@bind seed NumberField(0:1000, default=758))"

# ╔═╡ 4167489e-715b-4e62-8e56-3f2cd1317ccd
begin
	Random.seed!(seed)
	# sample code for a 2D array
	A_L = rand(Float64, L, L) .< 0.03
	A = zeros(n, n)

	for i in 1:n
		for j in 1:n
			A[i, j] = A_L[Int(ceil(i/(n/L))), Int(ceil(j/(n/L)))]
		end
	end
	
	
	# use show_image to display the array A
	# show_image(A)
	# show_image(A, [(0,0,0), (1,1,1)])
end

# ╔═╡ 0f0779fa-d610-429f-acd3-ac82b7842b14
begin
	Random.seed!(seed)
	alt_pos = rand(1:n, (altPs,2));
	alt_h = rand(Float64, (altPs,1))*max_height;
	hcat(alt_pos, alt_h)
	alt_p = hcat(alt_pos, alt_h);
	md"Generating random control points..."
end

# ╔═╡ b1538261-175d-4892-ab3d-2963f239b8df
alt_p

# ╔═╡ cb6482b5-c003-4ad2-8d8b-a60f3946b255
md"Power to raise the distance to control point... $(@bind power NumberField(0:1000; default=3))"

# ╔═╡ 8532f267-7e5f-45bb-8d82-6f86cfff7cc4
begin
	topo = zeros(Float64, n, n);
	topo = topography_gpu(topo, alt_p, power)
	md"Let's define the topography using the control points"
	# plotly()
	# show_image(topo, :grays)
end

# ╔═╡ 12351738-ddd3-4051-8880-504ecff343af
begin
	plotly()
	plot(1:n, 1:n,topo, st=:surface, ratio=1, zlim=[0,L], xlim=[0,n], ylim=[0,n],xlabel="X", ylabel="Y", zlabel="Z")
end

# ╔═╡ 3750d105-df07-4af7-9143-82b065fbb041
begin
	plotly()
	contour(1:n, 1:n,topo, levels=60, xlim=[0,n], ylim=[0,n], ratio=1, fill=true)
end

# ╔═╡ 9a877efd-b3cc-4d7e-ae9a-89d2e8a53356
md"Topography superposed with vegetation looks like this"

# ╔═╡ 2fff7da7-16ff-407d-92ef-24ee3469b9f4
begin
	plotly()
	surface_plot = plot(1:n, 1:n,plot_topo_gpu(topo, A), st=:surface, ratio=1, zlim=[0,L], xlim=[0,n], ylim=[0,n], xlabel="X", ylabel="Y", zlabel="Z")
end

# ╔═╡ 08c8c238-8a24-4743-aed5-0e2649758b61
md"### Slopes"

# ╔═╡ 81653527-a1fb-49ab-99db-5fdda6b669fd
md"""exploration radius, `e_radius = ` $(@bind e_radius NumberField(0:1000; default=3))"""

# ╔═╡ c8171ca3-c2d7-4220-b073-1ec76f559b25
md"""
The taylor series expansion of $f(x+h)$ at $h=0$ is,

$$f(x+h)=\frac{1}{24} h^4 f^{(4)}(x)+\frac{1}{6} h^3 f^{(3)}(x)+\frac{1}{2} h^2 f''(x)+h f'(x)+f(x)+O\left(h^5\right)$$

We can calculate the slope, $f'(x)$, at $x$ in the following manner,

$$\frac{f(x+h)-f(x-h)}{2 h} = f'(x)+\frac{1}{6} h^2 f^{(3)}(x)+O\left(h^4\right)$$

This is accurate with an error term $\propto h^2$. To improve, we use a neighbourhood of radius 2. That is, we use the fact that,

$$\frac{f(x+2h)-f(x-2 h)}{4 h}=f'(x)+\frac{2}{3} h^2 f^{(3)}(x)+O\left(h^4\right)$$

Like so,

$$\frac{1}{3} \left(4\cdot\frac{f(x+h)-f(x-h)}{2 h}-\frac{f(x+2h)-f(x-2 h)}{4 h}\right)=f'(x)-\frac{1}{30} h^4 f^{(5)}(x)+O\left(h^5\right)$$
"""

# ╔═╡ 15f17206-db9f-4896-9e32-93d025501917
begin
	function slope_kernel(A, Bx, By, m, n)
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
			Bx[i, j] = dfbydx/norm
			By[i, j] = dfbydy/norm
		elseif 2 <= i <= m-1 && 2 <= j <= n-1
			xph = A[i+1,j]
			xmh = A[i-1,j]

			yph = A[i,j+1]
			ymh = A[i,j-1]

			dfbydx = (xph-xmh)/2
			dfbydy = (yph-ymh)/2

			# B[i, j] = atan(dfbydy, dfbydx)
			norm = (dfbydx^2+dfbydy^2)^0.5
			Bx[i, j] = dfbydx/norm
			By[i, j] = dfbydy/norm
		elseif 1 <= i <= m && 1 <= j <= n
			Bx[i, j] = 0.0
			By[i, j] = 0.0
		end
		return
	end
	
	function slope_gpu(topo)
	    m, n = size(topo)
		topo_gpu = CuArray(topo)
		outp = fill((0.0, 0.0), n, n)
	    output_x = CuArray(fill(0.0, n, n))  
	    output_y = CuArray(fill(0.0, n, n))  
		threads_x = min(32, m)  # Limit to 32 threads in the x dimension
	    threads_y = min(32, n)  # Limit to 32 threads in the y dimension
	    blocks_x = ceil(Int, m / threads_x)
	    blocks_y = ceil(Int, n / threads_y)
		
	 	@cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) slope_kernel(topo_gpu, output_x, output_y, m, n)
	    
	    return collect(output_x), collect(output_y)
	end
	md"Kernel and method to generate topography slopes using central differences"
end

# ╔═╡ 230af3ed-9267-497c-a697-e422bcf04665
begin
	dx, dy = slope_gpu(topo);
	
	slope = [(dx[i, j], dy[i, j]) for i in 1:n, j in 1:n];
	md"Calculating the slope with a double central difference method"
end

# ╔═╡ c2a9fa1f-a405-4767-aec2-42196a70cc61
begin
	using DelimitedFiles;
	writedlm("slope.txt", slope);
	md"Let's write the slopes into a txt file for debugging"
end

# ╔═╡ 73014c35-ab99-47e2-bfcb-9076c0720bdf
md"## Enemies & Hill Climb... racing?"

# ╔═╡ daf19ff1-0012-4b12-b61f-1d9517178bf5
md"Let's first see how we can make our model realistically traverse the topography.

Since it's unlikely that a troop can climb any slope, we will try to make them move in the direction with the max feasible slope.

Let's deal with the following question: Should they do a random walk or should there be an \"ulterior\" motive? Time to explore!

What will a random walk look like?"

# ╔═╡ 5b8de4a5-f6d7-407a-8709-4e0d392e21b9
md"Set climbable slope to... $(@bind max_slope NumberField(1:10, default=7))%"

# ╔═╡ e9055da6-3c24-4fe9-919c-1040916c79c3
md"Let there be... $(@bind n_enem NumberField(1:10, default=3)) enemy clusters"

# ╔═╡ be20aaf3-473e-4be5-adcc-3db9eb3de213
begin
	Random.seed!(seed)
	enem_pos = rand(1:L, (n_enem,2));
	enem_z = [topo[row[1], row[2]] for row in eachrow(enem_pos)]
	enem_r = rand(1:3, (n_enem,1));
	enemies = hcat(enem_pos, enem_r);
	md"Generating random enemy clusters. The look like so..."
end

# ╔═╡ cb0bb5cd-a02b-457d-b47a-be623e8d50ed
enemies

# ╔═╡ 477ae165-07d6-4a64-8ce4-8c4b4c25011e
begin
	function neighbourhoods(radius, inc=0)
		n = []
		for r in 0:radius
			for i in 0:r
				if (inc!==0 || r !== 0)
					if(i !== 0) 
						push!(n, [-i, abs(r-i)], [i, abs(r-i)])
						if (r-i !== 0) 
							push!(n, [-i, -abs(r-i)], [i, -abs(r-i)])
						end
					else
						push!(n, [i, abs(r-i)])					
						if (r-i !== 0) 
							push!(n, [i, -abs(r-i)])
						end
					end
				end
			end					
		end
		return n
	end
	md"Definition of neighbourhood function that returns a von neumann neighbourhood set"
end

# ╔═╡ 86078a29-e2a6-470b-8757-b2efe2bf9eb8
md"Let's attempt to plot the enemies just like how we plotted bushes"

# ╔═╡ feb2345d-642a-4cd9-9d44-6ff2eb9f2ddd
begin

	topo_bush_enemies = plot_topo_gpu(topo, A)
	m, _ = size(enemies)
	for e in 1:m
		for nh in neighbourhoods(enemies[e, 3] * Int(n/L), 1)
			topo_bush_enemies[enemies[e, 1] * Int(n/L) + nh[1], enemies[e, 2] * Int(n/L) + nh[2]]+=2
		end
	end
	plot(1:n, 1:n,topo_bush_enemies, st=:surface, ratio=1, zlim=[0,L], xlim=[0,n], ylim=[0,n], xlabel="X", ylabel="Y", zlabel="Z")
end

# ╔═╡ c0bc8f94-9636-461a-9b34-fe0ccfefcb69
md"That doesn't look so great now, does it?

Let's plot the agents along with the bushes in a more beautiful manner. Green represents bushes and white for enemies."

# ╔═╡ a22d6084-18ed-4f71-886d-2ffc40ce599f
begin
	function gen_e_in_A(enemies, n, L)
		enemiesInA = zeros(n, n)
		for e in 1:size(enemies)[1]
			for nh in neighbourhoods(enemies[e, 3] * Int(n/L), 1)
				enemiesInA[enemies[e, 1] * Int(n/L) + nh[1], enemies[e, 2] * Int(n/L) + nh[2]] = 1
			end
		end
		return enemiesInA
	end
	enemiesInA = gen_e_in_A(enemies, n, L)
	
	function color(i, j, alt_ps, A, enemiesInA)
		z=0.0
		m, _ = size(alt_ps)
		norm = 0

		if(A[j, i]!=0)
			return -10
		elseif (enemiesInA[j, i]!=0)
			return max_height+10
		end
		for k in 1:m
			d = ((alt_ps[k, 1] - i)^2 + (alt_ps[k, 2] - j)^2)^0.5
			if (d > 0)
				z += alt_ps[k, 3]/d^power
				norm += 1/d^power
			else
				z = alt_ps[k, 3]
				return z
			end
		end
		z /= norm
		# println(typeof(z))
		return z
	end

	min_v = 10/(max_height+20)
	max_v = (max_height+10)/(max_height+20)
	custom_colorscale = [
	    (0.00, "#3bff00"),  # Green
	    (min_v - 0.000000001, "#3bff00"),  # Green
	    (min_v, "#222224"),  # Blue
	    (min_v + 1*(max_v-min_v)/5, "#3E2163"),  # Blue
		(min_v + 2*(max_v-min_v)/5, "#88236A"),# Yellow
		(min_v + 3*(max_v-min_v)/5, "#D04544"),# Yellow
		(min_v + 4*(max_v-min_v)/5, "#F78D1E"),# Yellow
		(max_v - 0.000000001, "#F1E760"),# Yellow
	    (max_v, "#ffffff"),  # Blue
	    (1.00, "#ffffff"),  # Blue
	]

	function colors_alias(x, y)
		return color(x, y, alt_p, A, enemiesInA)
	end
	
	x = 1:n
	y = 1:n
	
	surface(x = x, y = y, plot_topo_gpu(topo, A), colorscale=custom_colorscale, surfacecolor = colors_alias.(x', y), ratio=1, zlim=[0,L], xlim=[0,n], ylim=[0,n], xlabel="X", ylabel="Y", zlabel="Z")
end

# ╔═╡ 924c9d77-af8c-44b7-9053-b48aae4ad475
ENV["JULIA_CUDA_DEBUG"] = "2"

# ╔═╡ 9f30ffe2-6546-480b-a89d-0f557469e82d
begin
	function color_kernel(colors_A_gpu, alt_p_gpu, A_gpu, enemiesInA_gpu, m, n, max_height, power)
		i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
		j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
		if 1 <= i <= m && 1 <= j <= n
			colors_A_gpu[i, j]=0.0
			alt_ps_m, _ = size(alt_p_gpu)
			norm = 0
	
			if(A_gpu[i, j]!=0)
				colors_A_gpu[i, j] = -10
			elseif (enemiesInA_gpu[i, j]!=0)
				colors_A_gpu[i, j] = max_height+10
			else
				flag = 1
				for k in 1:alt_ps_m
					d = ((alt_p_gpu[k, 2] - i)^2 + (alt_p_gpu[k, 1] - j)^2)^0.5
					if (d > 0 && flag==1)
						colors_A_gpu[i, j] += alt_p_gpu[k, 3]/d^power
						norm += 1/d^power
					else
						colors_A_gpu[i, j] = alt_p_gpu[k, 3]
						flag = 0
					end
				end
				if(flag==1)
					colors_A_gpu[i, j] /= norm
				end
			end
		end
		return
	end
				
	function color_gpu(alt_p, A, enemiesInA, max_height, power)
		m, n = size(A)
		alt_p_gpu = CuArray(alt_p)
		A_gpu = CuArray(A)
		colors_A_gpu = similar(A_gpu)
		enemiesInA_gpu = CuArray(enemiesInA)  
		
		threads_x = min(25, m)  # Limit to 25 threads in the x dimension
		threads_y = min(25, n)  # Limit to 25 threads in the y dimension
		blocks_x = ceil(Int, m / threads_x)
		blocks_y = ceil(Int, n / threads_y)
		
		@cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) color_kernel(colors_A_gpu, alt_p_gpu, A_gpu, enemiesInA_gpu, m, n, max_height, power)
		
		return collect(colors_A_gpu)
	end
end

# ╔═╡ 282cd2e0-8b45-4625-af65-49f2167b1dc4
md"Clock $(@bind t Slider(1:100, show_value=true))"

# ╔═╡ 7382f5ff-0c87-4d1d-b45f-80286353135f
Markdown.parse("``t=$(t)\\ \\text{ticks}``")

# ╔═╡ a077d240-36e0-41cd-a4ff-f7e0ca62ca4e
md"Let's follow a \"gradient ascend\" method where the clusters just follow the direction with maximum ascend in hopes of reaching the peak."

# ╔═╡ fa304120-14f9-4c1a-a430-0438db6743f3
begin
	function gradient_ascend(enemies, t)
		enemiesAtT = copy(enemies)
		enemiesAtT_m, _ = size(enemiesAtT)
		surfacePlot = []
		for ti in 1:t
			for e in 1:enemiesAtT_m
				i, j = enemiesAtT[e, [1,2]]
				slopeHere = slope[i, j]
				r = enemiesAtT[e, 3]
				dx = ceil(slopeHere[1] * r)
				dy = ceil(slopeHere[2] * r)
				enemiesAtT[e, 1] = max(min(enemiesAtT[e, 1] + dx, L-r), r+1)
				enemiesAtT[e, 2] = max(min(enemiesAtT[e, 2] + dy, L-r), r+1)
			end
			enemiesInA = gen_e_in_A(enemiesAtT, n, L)
			
			function colors_alias2(x, y)
				return color(x, y, alt_p, A, enemiesInA)
			end
			
			x = 1:n
			y = 1:n
			
			surfacePlot = surface(x = x, y = y, plot_topo_gpu(topo, A), colorscale=custom_colorscale, surfacecolor = color_gpu(alt_p, A, enemiesInA, max_height, power), ratio=1, zlim=[0,L], xlim=[0,n], ylim=[0,n], xlabel="X", ylabel="Y", zlabel="Z", showscale=false)
		end
		return surfacePlot
	end
	surfacePlot = gradient_ascend(enemies, t)
	relayout!(surfacePlot,
		title="Mt Bruno Elevation",
		width=400, height=400,
		margin=attr(t=40, r=0, l=20, b=20)
	)
	
	camera = attr(
		eye=attr(x=0., y=0., z=40)
	)
	
	relayout!(surfacePlot, scene_camera=camera, title=name)
end

# ╔═╡ 67af10fc-c749-4042-be11-983f648f52ce
surfacePlot.layout[:scene]

# ╔═╡ fc5a9a4d-93bb-44de-8afa-99cfb6eac9e9
# ╠═╡ disabled = true
#=╠═╡
begin
	function random_climb(enemies, t)
		enemiesAtT = copy(enemies)
		enemiesAtT_m, _ = size(enemiesAtT)
		surfacePlot = []
		for ti in 1:t
			for e in 1:enemiesAtT_m
				i, j = enemiesAtT[e, [1,2]]
				slopeHere = slope[i, j]
				r = enemiesAtT[e, 3]
				dx = ceil(slopeHere[1] * r)
				dy = ceil(slopeHere[2] * r)
				enemiesAtT[e, 1] = max(min(enemiesAtT[e, 1] + dx, L-r), r+1)
				enemiesAtT[e, 2] = max(min(enemiesAtT[e, 2] + dy, L-r), r+1)
			end
			enemiesInA = gen_e_in_A(enemiesAtT, n, L)
			
			function colors_alias2(x, y)
				return color(x, y, alt_p, A, enemiesInA)
			end
			
			x = 1:n
			y = 1:n
			
			surfacePlot = surface(x = x, y = y, plot_topo_gpu(topo, A), colorscale=custom_colorscale, surfacecolor = color_gpu(alt_p, A, enemiesInA, max_height, power), ratio=1, zlim=[0,L], xlabel="X", ylabel="Y", zlabel="Z")
		end
		return surfacePlot
	end
	plot(random_climb(enemies, t))
end
  ╠═╡ =#

# ╔═╡ 0c4c7e4a-29e9-44c1-8392-ee688dc150b0
begin
	using PlutoPlotly, CSV, DataFrames, HTTP
	
	df = CSV.File(
	    HTTP.get("https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv").body
	) |> DataFrame
	
	p = plot(surface(z=Matrix{Float64}(df), showscale=false))
	relayout!(p,
	    title="Mt Bruno Elevation",
	    width=400, height=400,
	    margin=attr(t=40, r=0, l=20, b=20)
	)
	
	name = "eye = (x:0., y:0., z:2.5)"
	camera1 = attr(
	    eye=attr(x=0., y=0., z=2.5)
	)
	
	relayout!(p, scene_camera=camera1, title=name)
	p
end

# ╔═╡ 9083379c-842e-4f7c-936f-1f9e66861af0
begin
	# using Plots
	using PlutoPlotly
	using OffsetArrays
	using LaTeXStrings
	using PlutoUI
	using ColorTypes
	using CUDA
	CUDA.allowscalar(false)
	using Random
	md"Just Importing libraries here..."
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CSV = "~0.10.11"
CUDA = "~5.1.0"
ColorTypes = "~0.11.4"
DataFrames = "~1.6.1"
DelimitedFiles = "~1.9.1"
HTTP = "~1.10.0"
LaTeXStrings = "~1.3.1"
OffsetArrays = "~1.12.10"
PlotlyJS = "~0.18.11"
PlutoPlotly = "~0.4.3"
PlutoUI = "~0.7.53"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "2460f346b41cf26bd093768247615b28142af1a4"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AssetRegistry]]
deps = ["Distributed", "JSON", "Pidfile", "SHA", "Test"]
git-tree-sha1 = "b25e88db7944f98789130d7b503276bc34bc098e"
uuid = "bf4720bc-e11a-5d0c-854e-bdca1663c893"
version = "0.1.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "1c9b6f39f40dba0ef22244a175e2d4e42c8f6ee7"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.0"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Blink]]
deps = ["Base64", "Distributed", "HTTP", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Pkg", "Reexport", "Sockets", "WebIO"]
git-tree-sha1 = "b1c61fd7e757c7e5ca6521ef41df8d929f41e3af"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.8"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "Statistics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "64461b0e9df3069248979113ce8ab6d11bd371cf"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.1.0"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "1e42ef1bdb45487ff28de16182c0df4920181dc3"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.7.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "92394521ec4582c11d089a3b15b76ef2cb850994"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.10.0+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "85d7fb51afb3def5dcb85ad31c3707795c8bccc1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "9.1.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "a846f297ce9d09ccba02ead0cae70690e072a119"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.25.0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.Hiccup]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "6187bb2d5fcbb2007c39e7ac53308b0d371124bd"
uuid = "9fb69e20-1954-56bb-a84f-559cc56a8ff7"
version = "0.2.2"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSExpr]]
deps = ["JSON", "MacroTools", "Observables", "WebIO"]
git-tree-sha1 = "b413a73785b98474d8af24fd4c8a975e31df3658"
uuid = "97c1335a-c9c5-57fe-bc5d-ec35cebe8660"
version = "0.5.4"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "b0737cbbe1c8da6f1139d1c23e35e7cea129c0af"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.13"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "c879e47398a7ab671c782e02b51a4456794a7fa3"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.4.0"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "98eaee04d96d973e79c25d49167668c5c8fb50e2"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.27+1"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "f512dc13e64e96f703fd92ce617755ee6b5adf0f"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.8"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.Mustache]]
deps = ["Printf", "Tables"]
git-tree-sha1 = "a7cefa21a2ff993bff0456bf7521f46fc077ddf1"
uuid = "ffc61752-8dc7-55ee-8c37-f3e9cdd09e70"
version = "1.0.19"

[[deps.Mux]]
deps = ["AssetRegistry", "Base64", "HTTP", "Hiccup", "MbedTLS", "Pkg", "Sockets"]
git-tree-sha1 = "0bdaa479939d2a1f85e2f93e38fbccfcb73175a5"
uuid = "a975b10e-0019-58db-a62f-e48ff68538c9"
version = "1.0.1"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "8bc9ce4233be3c63f8dcd78ccaf1b63a9c0baa34"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "0.3.3"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ce3269ed42816bf18d500c9f63418d4b0d9f5a3b"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.1.0+2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyJS]]
deps = ["Base64", "Blink", "DelimitedFiles", "JSExpr", "JSON", "Kaleido_jll", "Markdown", "Pkg", "PlotlyBase", "REPL", "Reexport", "Requires", "WebIO"]
git-tree-sha1 = "3db9e7724e299684bf0ca8f245c0265c4bdd8dc6"
uuid = "f0f68f2c-4968-5e81-91da-67840de0976a"
version = "0.18.11"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "0b8880a45f96d8404ae1cf6e4d715e3a79369441"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.3"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8ec28846dbf846228a32de5a6912c63e2052e3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.53"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "3f43c2aae6aa4a2503b05587ab74f4f6aeff9fd0"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "5ef59aea6f18c25168842bded46b16662141ab87"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.7.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WebIO]]
deps = ["AssetRegistry", "Base64", "Distributed", "FunctionalCollections", "JSON", "Logging", "Observables", "Pkg", "Random", "Requires", "Sockets", "UUIDs", "WebSockets", "Widgets"]
git-tree-sha1 = "0eef0765186f7452e52236fa42ca8c9b3c11c6e3"
uuid = "0f1e0344-ec1d-5b48-a673-e5cf874b6c29"
version = "0.8.21"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "4162e95e05e79922e44b9952ccbc262832e4ad07"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.6.0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═4727903f-a54b-4d73-8998-fa99bb2481aa
# ╠═9083379c-842e-4f7c-936f-1f9e66861af0
# ╠═c8c9a170-7cc7-4bb3-b9dc-1654f4c2cefd
# ╟─df27f8a4-f258-43b4-acdc-b8ea0f9ffc88
# ╠═4167489e-715b-4e62-8e56-3f2cd1317ccd
# ╟─e5c741d7-7c52-4097-8d02-89d76495d53f
# ╟─29fb1a62-86bf-4bab-bb7e-dbbfd5024917
# ╠═7382f5ff-0c87-4d1d-b45f-80286353135f
# ╟─fd3512a7-8d52-4d25-9ad8-0cc80555da7f
# ╟─2a3753d3-c08c-4e85-907e-9ebb5a67dab3
# ╠═2fe91b37-1c3f-49ce-bfa2-702a180b78a0
# ╠═8327cfec-51df-4c38-839a-b7212ddb24e7
# ╟─701891a4-6a87-427e-af9b-487dec1dee4d
# ╟─0f344406-4816-4cd6-ae8e-83a8b918fa11
# ╟─4ec0a200-78df-4cfd-9efe-105dad6f4ef3
# ╟─fffa26a7-ecf6-4be0-ab7c-423665caf7a5
# ╟─72a7cb99-5483-4c82-9554-007c2ba44413
# ╠═cd4ee775-74d9-417f-9c97-6c8d321d7580
# ╟─0f0779fa-d610-429f-acd3-ac82b7842b14
# ╟─b1538261-175d-4892-ab3d-2963f239b8df
# ╠═ba6660df-59b7-4c70-b30f-b8548d63b0d2
# ╠═8532f267-7e5f-45bb-8d82-6f86cfff7cc4
# ╟─82d0e800-deb1-42fe-b1d3-2018d8639ff8
# ╟─8f0937f0-813b-4256-a8b9-afb22e092a42
# ╟─12351738-ddd3-4051-8880-504ecff343af
# ╟─6d4076dc-68c8-42f8-a43e-222e3410bdbf
# ╟─3750d105-df07-4af7-9143-82b065fbb041
# ╠═1add5389-3a8b-40b7-b999-8df22bb45900
# ╠═11f7bf70-4a39-451c-9bdb-9369742dcce0
# ╟─cb6482b5-c003-4ad2-8d8b-a60f3946b255
# ╟─9a877efd-b3cc-4d7e-ae9a-89d2e8a53356
# ╠═2fff7da7-16ff-407d-92ef-24ee3469b9f4
# ╟─08c8c238-8a24-4743-aed5-0e2649758b61
# ╟─81653527-a1fb-49ab-99db-5fdda6b669fd
# ╟─c8171ca3-c2d7-4220-b073-1ec76f559b25
# ╟─15f17206-db9f-4896-9e32-93d025501917
# ╠═230af3ed-9267-497c-a697-e422bcf04665
# ╟─c2a9fa1f-a405-4767-aec2-42196a70cc61
# ╟─73014c35-ab99-47e2-bfcb-9076c0720bdf
# ╟─daf19ff1-0012-4b12-b61f-1d9517178bf5
# ╟─5b8de4a5-f6d7-407a-8709-4e0d392e21b9
# ╟─e9055da6-3c24-4fe9-919c-1040916c79c3
# ╠═be20aaf3-473e-4be5-adcc-3db9eb3de213
# ╟─cb0bb5cd-a02b-457d-b47a-be623e8d50ed
# ╟─477ae165-07d6-4a64-8ce4-8c4b4c25011e
# ╟─86078a29-e2a6-470b-8757-b2efe2bf9eb8
# ╠═feb2345d-642a-4cd9-9d44-6ff2eb9f2ddd
# ╟─c0bc8f94-9636-461a-9b34-fe0ccfefcb69
# ╠═a22d6084-18ed-4f71-886d-2ffc40ce599f
# ╠═924c9d77-af8c-44b7-9053-b48aae4ad475
# ╠═9f30ffe2-6546-480b-a89d-0f557469e82d
# ╠═282cd2e0-8b45-4625-af65-49f2167b1dc4
# ╟─a077d240-36e0-41cd-a4ff-f7e0ca62ca4e
# ╠═fa304120-14f9-4c1a-a430-0438db6743f3
# ╠═67af10fc-c749-4042-be11-983f648f52ce
# ╠═fc5a9a4d-93bb-44de-8afa-99cfb6eac9e9
# ╠═0c4c7e4a-29e9-44c1-8392-ee688dc150b0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
