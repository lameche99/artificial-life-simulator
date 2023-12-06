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

# ╔═╡ 9083379c-842e-4f7c-936f-1f9e66861af0
begin
	using Plots
	using PlotlyJS
	using OffsetArrays
	using LaTeXStrings
	using PlutoUI
	using ColorTypes
	using CUDA
	using GPUArrays
	CUDA.allowscalar(false)
	using Random
	md"Just Importing libraries here..."
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
md"Random Seed, $(@bind seed NumberField(0:1000, default=809))"

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
	show_image(A)
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
	plot(1:n, 1:n,topo, st=:surface, ratio=1, zlim=[0,L],xlabel="X", ylabel="Y", zlabel="Z")
end

# ╔═╡ 3750d105-df07-4af7-9143-82b065fbb041
begin
	plotly()
	contour(1:n, 1:n,topo, levels=60, fill=true)
end

# ╔═╡ 9a877efd-b3cc-4d7e-ae9a-89d2e8a53356
md"Topography superposed with vegetation looks like this"

# ╔═╡ 2fff7da7-16ff-407d-92ef-24ee3469b9f4
begin
	plotly()
	surface_plot = plot(1:n, 1:n,plot_topo_gpu(topo, A), st=:surface, ratio=1, zlim=[0,L],xlabel="X", ylabel="Y", zlabel="Z")
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
	plot(1:n, 1:n,topo_bush_enemies, st=:surface, ratio=1, zlim=[0,L],xlabel="X", ylabel="Y", zlabel="Z")
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
	
	surface(x = x, y = y, plot_topo_gpu(topo, A), colorscale=custom_colorscale, surfacecolor = colors_alias.(x', y), ratio=1, zlim=[0,L], xlabel="X", ylabel="Y", zlabel="Z")
end

# ╔═╡ 282cd2e0-8b45-4625-af65-49f2167b1dc4
md"Clock $(@bind t Slider(1:100, show_value=true))"

# ╔═╡ 7382f5ff-0c87-4d1d-b45f-80286353135f
Markdown.parse("``t=$(t)\\ \\text{ticks}``")

# ╔═╡ 924c9d77-af8c-44b7-9053-b48aae4ad475
ENV["JULIA_CUDA_DEBUG"] = "2"

# ╔═╡ fa304120-14f9-4c1a-a430-0438db6743f3
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
				
				threads_x = min(32, m)  # Limit to 32 threads in the x dimension
			    threads_y = min(32, n)  # Limit to 32 threads in the y dimension
			    blocks_x = ceil(Int, m / threads_x)
			    blocks_y = ceil(Int, n / threads_y)
				
			 	@cuda threads=(threads_x, threads_y) blocks=(blocks_x, blocks_y) color_kernel(colors_A_gpu, alt_p_gpu, A_gpu, enemiesInA_gpu, m, n, max_height, power)
			    
			    return collect(colors_A_gpu)
			end
			
			x = 1:n
			y = 1:n
			
			surfacePlot = surface(x = x, y = y, plot_topo_gpu(topo, A), colorscale=custom_colorscale, surfacecolor = color_gpu(alt_p, A, enemiesInA, max_height, power), ratio=1, zlim=[0,L], xlabel="X", ylabel="Y", zlabel="Z")
		end
		return surfacePlot
	end
	# Set camera attributes to achieve a top view
	layout = Layout(
	    scene = Dict(
	        "camera" => Dict(
	            "eye" => Dict("x" => 0, "y" => 0, "z" => 2),  # Set the camera position
	            "center" => Dict("x" => 0, "y" => 0, "z" => 0)  # Set the center point to look at
	        )
	    )
	)

	plot(random_climb(enemies, t), layout)
end

# ╔═╡ fc5a9a4d-93bb-44de-8afa-99cfb6eac9e9


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ColorTypes = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CUDA = "~5.0.0"
ColorTypes = "~0.11.4"
DelimitedFiles = "~1.9.1"
GPUArrays = "~9.1.0"
LaTeXStrings = "~1.3.0"
OffsetArrays = "~1.12.10"
PlotlyJS = "~0.18.11"
Plots = "~1.39.0"
PlutoUI = "~0.7.52"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "d1a510a0fcbe407e61e07ab82c708863ad0a70a7"

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
git-tree-sha1 = "68c4c187a232e7abe00ac29e3b03e09af9d77317"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.0"
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

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Blink]]
deps = ["Base64", "Distributed", "HTTP", "JSExpr", "JSON", "Lazy", "Logging", "MacroTools", "Mustache", "Mux", "Pkg", "Reexport", "Sockets", "WebIO"]
git-tree-sha1 = "b1c61fd7e757c7e5ca6521ef41df8d929f41e3af"
uuid = "ad839575-38b3-5650-b840-f874b8c74a25"
version = "0.12.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "Statistics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "f062a48c26ae027f70c44f48f244862aec47bf99"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.0.0"

    [deps.CUDA.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2f64185414751a5f878c4ab616c0edd94ade3419"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.6.0+4"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "a38bc81bd6fdc7334de537aec3ae60e7b098daa2"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.9.2+4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

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
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

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

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionalCollections]]
deps = ["Test"]
git-tree-sha1 = "04cb9cfaa6ba5311973994fe3496ddec19b6292a"
uuid = "de31a74c-ac4f-5751-b3fd-e18cd04993ca"
version = "0.5.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

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
git-tree-sha1 = "5e4487558477f191c043166f8301dd0b4be4e2b2"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.24.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5eab648309e2e060198b45820af1a37182de3cce"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

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
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

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

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "9fb0b890adab1c0a4a475d4210d51f228bfc250d"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.6"

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

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

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
git-tree-sha1 = "5f1ecfddb6abde48563d08b2cc7e5116ebcd6c27"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.10"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "4ea2928a96acfcf8589e6cd1429eff2a3a82c366"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.3.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "e7c01b69bcbcb93fd4cbc3d0fea7d229541e18d2"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.26+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

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

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

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

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

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

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

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
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6842ce83a836fbbc0cfeca0b5a4de1a4dcbdb8d1"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.8"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "7c29f0e8c575428bd84dc3c72ece5178caa67336"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.2+2"

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

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

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
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

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
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"
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

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

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
git-tree-sha1 = "49cbf7c74fafaed4c529d47d48c8f7da6a19eb75"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.1"
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

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

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

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "47cf33e62e138b920039e8ff9f9841aafe1b733e"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.35.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
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
# ╟─ba6660df-59b7-4c70-b30f-b8548d63b0d2
# ╠═8532f267-7e5f-45bb-8d82-6f86cfff7cc4
# ╟─82d0e800-deb1-42fe-b1d3-2018d8639ff8
# ╟─8f0937f0-813b-4256-a8b9-afb22e092a42
# ╟─12351738-ddd3-4051-8880-504ecff343af
# ╟─6d4076dc-68c8-42f8-a43e-222e3410bdbf
# ╟─3750d105-df07-4af7-9143-82b065fbb041
# ╠═1add5389-3a8b-40b7-b999-8df22bb45900
# ╟─11f7bf70-4a39-451c-9bdb-9369742dcce0
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
# ╟─2351b2a4-c0d0-4769-ab1b-be778212bee9
# ╟─477ae165-07d6-4a64-8ce4-8c4b4c25011e
# ╟─86078a29-e2a6-470b-8757-b2efe2bf9eb8
# ╠═feb2345d-642a-4cd9-9d44-6ff2eb9f2ddd
# ╟─c0bc8f94-9636-461a-9b34-fe0ccfefcb69
# ╠═a22d6084-18ed-4f71-886d-2ffc40ce599f
# ╠═b32adad4-c88b-4c19-98d1-2eaa2454b1fe
# ╠═282cd2e0-8b45-4625-af65-49f2167b1dc4
# ╠═924c9d77-af8c-44b7-9053-b48aae4ad475
# ╠═fa304120-14f9-4c1a-a430-0438db6743f3
# ╠═fc5a9a4d-93bb-44de-8afa-99cfb6eac9e9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
