alias build='cargo build --release'
alias run='./target/release/smooth-image'

build

# problem (e)

set -x #echo on

echo Running CG Algorithm

run in-paint \
  -i test/test_images/512_512_pens.png \
  -m test/test_masks/640_640_scratches_01.png \
  -o test/test_outputs/problem-e-pens-cg.png \
  --algo cg \
  --init zero \
  --tol 0.0001 \
  --mu 0.01

run in-paint \
  -i test/test_images/1024_1024_bluestreet.png \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-bluestree-cg.png \
  --algo cg \
  --init random \
  --tol 0.0001 \
  --mu 0.01

run in-paint \
  -i test/test_images/4096_4096_husky.jpg \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-husky-cg.png \
  --algo cg \
  --init random \
  --tol 0.0001 \
  --mu 0.01

echo Running AG Algorithm

run in-paint \
  -i test/test_images/512_512_pens.png \
  -m test/test_masks/640_640_scratches_01.png \
  -o test/test_outputs/problem-e-pens-ag.png \
  --algo ag \
  --init zero \
  --tol 0.0001 \
  --mu 0.01

run in-paint \
  -i test/test_images/1024_1024_bluestreet.png \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-bluestree-ag.png \
  --algo ag \
  --init random \
  --tol 0.0001 \
  --mu 0.01

run in-paint \
  -i test/test_images/4096_4096_husky.jpg \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-husky-ag.png \
  --algo ag \
  --init random \
  --tol 0.0001 \
  --mu 0.01
