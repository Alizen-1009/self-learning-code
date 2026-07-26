#include <iostream>

#include "cute/tensor.hpp"

int main() {
    using namespace cute;

    // CuTe starts from layouts: a coordinate -> a linear memory index.
    // This layout describes a row-major 4 x 8 matrix.
    auto row_major = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                 make_stride(Int<8>{}, Int<1>{}));

    std::cout << "row_major layout:\n";
    print(row_major);
    std::cout << "\n\nvisualized indices:\n";
    print_layout(row_major);

    auto coord = make_coord(2, 3);
    std::cout << "\nlinear index of coordinate (2, 3): "
              << row_major(coord) << "\n";

    // A tile is just another layout object. Higher-level CUTLASS kernels use
    // these layout objects to describe how blocks/warps/threads own data.
    auto tile = make_layout(make_shape(Int<2>{}, Int<4>{}),
                            make_stride(Int<4>{}, Int<1>{}));
    std::cout << "\n2 x 4 tile layout:\n";
    print_layout(tile);

    return 0;
}
