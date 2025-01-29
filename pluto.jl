using Pluto

# Get target directory from command line arguments or use script's directory
target_dir = if !isempty(ARGS)
    Base.Filesystem.expanduser(ARGS[1])  # Handles ~/ paths
else
    @__DIR__  # Directory where this script resides
end

# Change working directory and start Pluto
println("Starting Pluto in: ", target_dir)
cd(target_dir)
Pluto.run()