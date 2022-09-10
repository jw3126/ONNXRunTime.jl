using TOML, Artifacts, Base.BinaryPlatforms

import ONNXRuntime_jll

include("platform_augmentation.jl")

artifacts_toml = Artifacts.find_artifacts_toml(pathof(ONNXRuntime_jll))

# Get "target triplet" from ARGS, if given (defaulting to the host triplet otherwise)
target_triplet = get(ARGS, 1, Base.BinaryPlatforms.host_triplet())

# Augment this platform object with any special tags we require
platform = augment_platform!(HostPlatform(parse(Platform, target_triplet)))

# Select all downloadable artifacts that match that platform
artifacts = select_downloadable_artifacts(artifacts_toml; platform)

# Output the result to `stdout` as a TOML dictionary
TOML.print(stdout, artifacts)
