#include <utility>
#include <variant>
#include <vector>

// 一个元素可以是整数，也可以是由同类元素组成的数组。
struct Value {
    using Array = std::vector<Value>;
    std::variant<int, Array> data;

    Value(int value) : data(value) {}
    Value(Array array) : data(std::move(array)) {}
};

using Array = Value::Array;

void flat_into(const Array& array, int depth, Array& result) {
    for (const Value& item : array) {
        const Array* nested = std::get_if<Array>(&item.data);
        if (nested != nullptr && depth > 0)
            flat_into(*nested, depth - 1, result);
        else
            result.push_back(item);
    }
}

// depth=0 不展开，depth=1 展开一层；保持顺序，不修改原数组。
Array flat(const Array& array, int depth = 1) {
    Array result;
    flat_into(array, depth, result);
    return result;
}

// 使用示例（C++17）：
// Array a = {Array{Array{1, 2}, Array{3}}, Array{Array{4, 5}}};
// Array b = flat(a, 1);  // [[1, 2], [3], [4, 5]]
// Array c = flat(a, 2);  // [1, 2, 3, 4, 5]
