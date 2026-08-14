#!/usr/bin/env julia
#
# Fully automated update of the ONNXRunTime.jl artifacts.
#
#     julia update_artifacts.jl [version] [--force]
#
# See README.md in this directory for what it does and what it needs.

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using ArtifactUtils: add_artifact!
using Base.BinaryPlatforms: Platform
using CodecZlib: GzipCompressorStream
using Downloads: Downloads
using Tar: Tar
using p7zip_jll: p7zip

# The version to update to, unless one is passed on the command line.
const DEFAULT_VERSION = v"1.20.1"

# Repository the repacked tarballs are released to. Upstream ships the
# Windows builds as zip files, but Artifacts.jl only understands tarballs,
# so those have to be repacked and rehosted by us.
const ARTIFACTS_REPO = "jw3126/ONNXRunTimeArtifacts"

const PKG_ROOT = dirname(@__DIR__)
const WORKDIR = joinpath(@__DIR__, "build")

"""
    artifact_items(version)

The full artifact matrix: one entry per (artifact, platform) pair that ends up
in `Artifacts.toml`. `download_name` is the asset name in the upstream
onnxruntime release; `.zip` entries are repacked and rehosted by us.
"""
function artifact_items(version)
    [
        (artifact_name = "onnxruntime_cpu",
         download_name = "onnxruntime-win-x64-$version.zip",
         platform = Platform("x86_64", "windows")),
        (artifact_name = "onnxruntime_gpu",
         download_name = "onnxruntime-win-x64-gpu-$version.zip",
         platform = Platform("x86_64", "windows")),
        (artifact_name = "onnxruntime_cpu",
         download_name = "onnxruntime-linux-x64-$version.tgz",
         platform = Platform("x86_64", "linux")),
        (artifact_name = "onnxruntime_gpu",
         download_name = "onnxruntime-linux-x64-gpu-$version.tgz",
         platform = Platform("x86_64", "linux")),
        (artifact_name = "onnxruntime_cpu",
         download_name = "onnxruntime-linux-aarch64-$version.tgz",
         platform = Platform("aarch64", "linux")),
        (artifact_name = "onnxruntime_cpu",
         download_name = "onnxruntime-osx-universal2-$version.tgz",
         platform = Platform("x86_64", "macos")),
        (artifact_name = "onnxruntime_cpu",
         download_name = "onnxruntime-osx-arm64-$version.tgz",
         platform = Platform("aarch64", "macos")),
    ]
end

needs_repack(item) = endswith(item.download_name, ".zip")

repacked_name(item) = replace(item.download_name, r"\.zip$" => ".tgz")

release_tag(version) = "v$version-rc1"

function upstream_url(item, version)
    "https://github.com/microsoft/onnxruntime/releases/download/v$version/$(item.download_name)"
end

function rehosted_url(item, version)
    "https://github.com/$ARTIFACTS_REPO/releases/download/$(release_tag(version))/$(repacked_name(item))"
end

artifact_url(item, version) =
    needs_repack(item) ? rehosted_url(item, version) : upstream_url(item, version)

# ---------------------------------------------------------------------------
# Step 1: download and repack the Windows zips
# ---------------------------------------------------------------------------

function download_cached(url, path)
    isfile(path) && return path
    @info "Downloading" url path
    mkpath(dirname(path))
    tmp = path * ".part"
    Downloads.download(url, tmp)
    mv(tmp, path, force = true)
    return path
end

"""
    repack(item, version, workdir)

Download the upstream zip of `item` and repack it as a `.tgz`, preserving the
single top level directory that `src/capi.jl` relies on. All files are made
executable because Windows `dlopen` refuses to load a dll otherwise, see
https://github.com/JuliaLang/julia/issues/38993
"""
function repack(item, version, workdir)
    tarball = joinpath(workdir, repacked_name(item))
    isfile(tarball) && return tarball

    zipfile = download_cached(upstream_url(item, version), joinpath(workdir, item.download_name))
    stage = joinpath(workdir, "stage", replace(item.download_name, r"\.zip$" => ""))
    rm(stage, recursive = true, force = true)
    mkpath(stage)

    @info "Repacking" zipfile tarball
    run(`$(p7zip()) x -y -bso0 -o$stage $zipfile`)
    chmod(stage, 0o755, recursive = true)

    entries = readdir(stage)
    # `make_lib!` in src/capi.jl does `only(readdir(artifact_path(h)))`.
    length(entries) == 1 || error("Expected a single top level directory in $zipfile, got $entries")

    tmp = tarball * ".part"
    open(tmp, "w") do raw
        gz = GzipCompressorStream(raw)
        try
            Tar.create(stage, gz)
        finally
            close(gz)
        end
    end
    mv(tmp, tarball, force = true)
    return tarball
end

# ---------------------------------------------------------------------------
# Step 2: publish the repacked tarballs as a GitHub release
# ---------------------------------------------------------------------------

function check_gh()
    isnothing(Sys.which("gh")) && error("The GitHub CLI `gh` is required, see https://cli.github.com")
    success(pipeline(`gh auth status`, stdout = devnull, stderr = devnull)) ||
        error("`gh` is not authenticated, run `gh auth login` (needs the `repo` scope).")
end

"""
    release_assets(tag)

Names of the assets attached to release `tag`, or `nothing` if there is no such
release yet.
"""
function release_assets(tag)
    jq = ".assets[].name"
    cmd = `gh release view $tag --repo $ARTIFACTS_REPO --json assets --jq $jq`
    out = IOBuffer()
    success(pipeline(cmd, stdout = out, stderr = devnull)) || return nothing
    return split(String(take!(out)), '\n', keepempty = false)
end

function publish_release(version, tarballs; exists::Bool, force::Bool = false)
    tag = release_tag(version)
    if !exists
        isempty(tarballs) && return nothing
        notes = "Windows builds of onnxruntime $version, repacked from the upstream " *
                "zip files as tarballs for Artifacts.jl."
        @info "Creating release" tag assets = basename.(tarballs)
        run(`gh release create $tag --repo $ARTIFACTS_REPO --title $tag --notes $notes $tarballs`)
    elseif isempty(tarballs)
        @info "Release already has every asset, nothing to upload" tag
    else
        @info "Uploading assets" tag assets = basename.(tarballs)
        flags = force ? ["--clobber"] : String[]
        run(`gh release upload $tag --repo $ARTIFACTS_REPO $flags $tarballs`)
    end
    return nothing
end

"""
    wait_for_url(url; timeout, interval)

GitHub needs a moment before a freshly uploaded asset is downloadable. Poll
until it is, so that the `Artifacts.toml` step does not race the upload.
"""
function wait_for_url(url; timeout = 300, interval = 5)
    deadline = time() + timeout
    while true
        response = Downloads.request(url, method = "HEAD", throw = false)
        if response isa Downloads.Response && response.status == 200
            return nothing
        end
        time() > deadline && error("Timed out waiting for $url to become available")
        @info "Waiting for asset to become available" url
        sleep(interval)
    end
end

# ---------------------------------------------------------------------------
# Step 3: regenerate Artifacts.toml and bump src/versions.jl
# ---------------------------------------------------------------------------

"""
    write_artifacts_toml(version, dest)

Build `Artifacts.toml` from scratch in a temporary directory and move it into
place. Building from scratch means platforms that were dropped from
`artifact_items` do not linger as stale entries.
"""
function write_artifacts_toml(version, dest)
    tmp = joinpath(mktempdir(), "Artifacts.toml")
    for item in artifact_items(version)
        url = artifact_url(item, version)
        @info "Adding artifact" item.artifact_name url
        add_artifact!(tmp, item.artifact_name, url,
                      force = true, platform = item.platform, lazy = true)
    end
    mv(tmp, dest, force = true)
    @info "Wrote" dest
    return dest
end

function update_versions_jl(version, path = joinpath(PKG_ROOT, "src", "versions.jl"))
    old = read(path, String)
    new = replace(old, r"^const onnxruntime_version = v\"[^\"]*\"$"m =>
                       "const onnxruntime_version = v\"$version\"")
    occursin("v\"$version\"", new) ||
        error("Could not set onnxruntime_version in $path, please check the file.")
    if new != old
        write(path, new)
        @info "Bumped onnxruntime_version" path version
    end
    return path
end

# ---------------------------------------------------------------------------

function main(version = DEFAULT_VERSION; workdir = WORKDIR, force = false)
    version = VersionNumber(version)
    items = artifact_items(version)
    check_gh()
    mkpath(workdir)

    tag = release_tag(version)
    attached = release_assets(tag)   # nothing if the release does not exist yet
    present = (force || isnothing(attached)) ? String[] : attached

    # Anything already attached to the release needs neither a download nor a
    # repack. `Artifacts.toml` is generated from the released assets either way,
    # so skipping cannot make it disagree with what is published.
    todo = [item for item in items if needs_repack(item) && !(repacked_name(item) in present)]
    skipped = [repacked_name(item) for item in items
               if needs_repack(item) && repacked_name(item) in present]
    isempty(skipped) || @info "Already attached to $tag, skipping" assets = skipped

    tarballs = [repack(item, version, workdir) for item in todo]
    publish_release(version, tarballs; exists = !isnothing(attached), force)
    for item in items
        needs_repack(item) && wait_for_url(rehosted_url(item, version))
    end

    write_artifacts_toml(version, joinpath(PKG_ROOT, "Artifacts.toml"))
    update_versions_jl(version)

    @info """
    Done. Remaining manual steps:
      * run the test suite
      * bump the ONNXRunTime version in Project.toml
      * commit Artifacts.toml and src/versions.jl
    """
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    force = "--force" in ARGS
    positional = filter(!startswith("-"), ARGS)
    main(isempty(positional) ? DEFAULT_VERSION : only(positional); force)
end
