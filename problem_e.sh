build='cargo build --release'
run='./target/release/smooth-image'

$build || exit

# problem (e)

set -x #echo on

echo Running CG Algorithm

$run in-paint \
  -i test/test_images/512_512_stars_01.png \
  -m test/test_masks/640_640_scratches_02.png \
  -o test/test_outputs/problem-e-stars-cg.png \
  --algo cg \
  --init zero \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 10 \
  --mono

$run in-paint \
  -i test/test_images/1024_1024_books.jpg \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-books-cg.png \
  --algo cg \
  --init zero \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 10 \
  --mono

$run in-paint \
  -i test/test_images/4096_4096_husky.jpg \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-husky-cg.png \
  --algo cg \
  --init zero \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 10 \
  --mono

echo Running AG Algorithm

$run in-paint \
  -i test/test_images/512_512_stars_01.png \
  -m test/test_masks/640_640_scratches_02.png \
  -o test/test_outputs/problem-e-stars-ag.png \
  --algo ag \
  --init zero \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 100 \
  --mono

$run in-paint \
  -i test/test_images/1024_1024_books.jpg \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-books-ag.png \
  --algo ag \
  --init zero \
  --tol 0.0001 \
  --mu 0.01 \
  --metric-step 100 \
  --mono

$run in-paint \
  -i test/test_images/4096_4096_husky.jpg \
  -m test/test_masks/512_512_handwriting.png \
  -o test/test_outputs/problem-e-husky-ag.png \
  --algo ag \
  --init zero \
  --tol 0.001 \
  --mu 0.01 \
  --metric-step 100 \
  --mono
