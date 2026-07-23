# OpenFOAM Case — Supersonic Wedge (M=2.5, θ=15°)

## Solver
`rhoCentralFoam` — density-based compressible solver using the
Kurganov-Tadmor central scheme. Correct for supersonic inviscid flow.

## How to run

```bash
cd openfoam_case

# 1. Generate the 2-D wedge mesh (requires blockMesh)
blockMesh

# 2. Check the mesh
checkMesh

# 3. Run the solver
rhoCentralFoam > log.rhoCentralFoam 2>&1 &

# 4. Monitor convergence
tail -f log.rhoCentralFoam

# 5. Post-process surfaces (generates the .raw files the PINN reads)
postProcess -func surfaces
```

## Mesh setup (blockMeshDict)
You need to create `system/blockMeshDict` defining the wedge geometry.
A 15° wedge in a 2m × 1.2m domain with ~200×100 cells is sufficient.

Key blocks:
- Inlet face at x=0
- Wedge surface along y = x·tan(15°)
- Outlet at x=2m
- Top boundary at y=1.2m

## Output location
After running postProcess, data appears in:
```
postProcessing/surfaces/{time}/
    wedge_plane_p.raw
    wedge_plane_U.raw
    wedge_plane_rho.raw
    wedge_plane_T.raw
    wedge_plane_Ma.raw
```

Rename these to match the loader:
```bash
cd postProcessing/surfaces/{time}/
mv wedge_plane_p.raw   wedge_p.raw
mv wedge_plane_U.raw   wedge_U.raw
mv wedge_plane_rho.raw wedge_rho.raw
mv wedge_plane_T.raw   wedge_T.raw
mv wedge_plane_Ma.raw  wedge_Ma.raw
```

Then run the PINN:
```bash
cd ../../../..
python main.py --mode openfoam --data_dir postProcessing/surfaces
```

## Expected results
- Shock angle β ≈ 36.9° (theory)
- Post-shock Mach ≈ 1.87
- Post-shock pressure ≈ 250,000 Pa
- Post-shock temperature ≈ 397 K
