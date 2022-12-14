xcrun --sdk macosx metal -ffast-math metal/ag_method.metal -c -o ./target/ag_method.air || exit
xcrun --sdk macosx metal -ffast-math metal/cg_method.metal -c -o ./target/cg_method.air || exit
xcrun --sdk macosx metallib ./target/ag_method.air ./target/cg_method.air -o ./metallib/gpu.metallib
