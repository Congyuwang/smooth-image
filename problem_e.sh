build='cargo build --release'
run='./target/release/smooth-image'

$build

# problem (e)

set -x #echo on

echo Running CG Algorithm

$run in-paint \
  -i test/test_images/512_512_pens.png \
  -m test/test_masks/640_640_scratches_02.png \
  -o test/test_outputs/problem-e-pens-cg.png \
  --algo cg \
  --init zero \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 10

$run in-paint \
  -i test/test_images/1024_1024_bluestreet.png \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-bluestreet-cg.png \
  --algo cg \
  --init random \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 10

$run in-paint \
  -i test/test_images/4096_4096_husky.jpg \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-husky-cg.png \
  --algo cg \
  --init random \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 10

echo Running AG Algorithm

$run in-paint \
  -i test/test_images/512_512_pens.png \
  -m test/test_masks/640_640_scratches_02.png \
  -o test/test_outputs/problem-e-pens-ag.png \
  --algo ag \
  --init zero \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 10

$run in-paint \
  -i test/test_images/1024_1024_bluestreet.png \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-bluestreet-ag.png \
  --algo ag \
  --init random \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 10

$run in-paint \
  -i test/test_images/4096_4096_husky.jpg \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-husky-ag.png \
  --algo ag \
  --init random \
  --tol 0.001 \
  --mu 0.01 \
  --metric-step 10
