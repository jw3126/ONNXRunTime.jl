# Updating the artifacts

To update ONNXRunTime.jl to a new onnxruntime release, set `DEFAULT_VERSION` in
`update_artifacts.jl` and run

```
julia --project=. update_artifacts.jl
```

or pass the version explicitly, without editing anything:

```
julia --project=. update_artifacts.jl 1.20.1
```

That is the whole update flow. It

1. downloads the upstream release assets,
2. repacks the Windows zip files as tarballs, because Artifacts.jl only
   understands tarballs,
3. publishes those tarballs as a release on
   [ONNXRunTimeArtifacts](https://github.com/jw3126/ONNXRunTimeArtifacts),
   skipping anything already attached to that release,
4. waits until the uploaded assets are downloadable,
5. regenerates `../Artifacts.toml` from scratch,
6. bumps `onnxruntime_version` in `../src/versions.jl`.

Afterwards, run the test suite, bump the package version in `../Project.toml`
and commit `Artifacts.toml` and `src/versions.jl`.

## Requirements

* The [GitHub CLI](https://cli.github.com) `gh`, authenticated with the `repo`
  scope for `jw3126/ONNXRunTimeArtifacts` (`gh auth login`). This is the only
  step that needs credentials.
* Everything else is handled by this project's dependencies.

## Notes

* Downloads and repacked tarballs are cached in `build/`, which is gitignored.
  Delete it to force a fresh download.
* The script is idempotent and cheap to rerun. Assets that are already attached
  to the release are skipped entirely, so they are not downloaded, not repacked
  and not uploaded again. A rerun on an unchanged version does no network work
  beyond regenerating `Artifacts.toml`.
* Skipping cannot desynchronize anything: `Artifacts.toml` is always generated
  from the assets as published, never from the local tarballs.
* To replace assets that are already attached, for instance after a botched
  upload, pass `--force`. That repacks them and uploads with `--clobber`.
* Only platforms listed in `artifact_items` end up in `Artifacts.toml`. Since
  the file is rebuilt from scratch, removing a platform there removes it from
  `Artifacts.toml` too.
* All files in the repacked Windows tarballs are made executable. Windows
  `dlopen` refuses to load a dll otherwise, see
  [JuliaLang/julia#38993](https://github.com/JuliaLang/julia/issues/38993).
* `src/capi.jl` expects each artifact to contain exactly one top level
  directory, holding `lib/`. The repacking preserves that and errors out if
  upstream ever changes the layout.
