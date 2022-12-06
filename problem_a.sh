build='cargo build --release'
run='./target/release/smooth-image'

$build

# problem (a)
$run mask-image \
  -i ./test/test_images/1024_1024_bluestreet.png \
  -m ./test/test_masks/640_640_handwriting.png \
  -o ./test/test_outputs/problem-a-output-1.png

$run mask-image \
  -i ./test/test_images/1024_1024_mountain.jpg \
  -m ./test/test_masks/640_640_scratches_01.png \
  -o ./test/test_outputs/problem-a-output-2.png
