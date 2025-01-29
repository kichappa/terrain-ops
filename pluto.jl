using Pluto

# Get target directory from command line arguments or use script's directory
target_dir = if !isempty(ARGS) && isdir(ARGS[1])
	Base.Filesystem.expanduser(ARGS[1])  # Handles ~/ paths
else
	@__DIR__  # Directory where this script resides
end

# Determine the notebook file to open
notebook_file = if !isempty(ARGS) && endswith(ARGS[end], ".jl")
	ARGS[end]  # Use the provided notebook file
else
	"notebook.jl"  # Default notebook file
end

# Construct the full path to the notebook
notebook_path = joinpath(target_dir, notebook_file)

# Check if the notebook file exists, or create it if it doesn't
if !isfile(notebook_path)
	println("Creating new notebook: ", notebook_path)
	touch(notebook_path)
end

# Change working directory and start Pluto with the notebook
println("Starting Pluto in: ", target_dir)
println("Opening notebook: ", notebook_path)
cd(target_dir)
Pluto.run(notebook = notebook_path)
