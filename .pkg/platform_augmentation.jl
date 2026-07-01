using Libdl, Base.BinaryPlatforms

function augment_platform!(p::Platform, tag::Union{String,Nothing} = nothing)
    if tag === nothing
        return p
    end

    if tag === "cuda"
        if haskey(p, tag)
            return p
        end

        # Open libcuda explicitly, so it gets `dlclose()`'ed after we're done
        try
            dlopen("libcuda") do lib
                # find symbol to ask for driver version; if we can't find it, just silently continue
                cuDriverGetVersion = dlsym(lib, "cuDriverGetVersion"; throw_error=false)
                if cuDriverGetVersion !== nothing
                    # Interrogate CUDA driver for driver version:
                    driverVersion = Ref{Cint}()
                    ccall(cuDriverGetVersion, UInt32, (Ptr{Cint},), driverVersion)

                    # Store only the major version
                    p[tag] = div(driverVersion, 1000)
                end
            end
        catch
        end

        return p
    else
        @warn "Unexpected tag $tag for $p"
        return p
    end
end
