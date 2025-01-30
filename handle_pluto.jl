function is_running_in_pluto()
	# Check environment variable
	if get(ENV, "PLUTO_PROJECT", "") != ""
		return true
	end

	# Check call stack
	for stack in stacktrace()
		if occursin("Pluto", string(stack))
			return true
		end
	end

	return false
end