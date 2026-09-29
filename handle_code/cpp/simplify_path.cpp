/*
题目：规范化 Linux 绝对路径
【题意】路径以 '/' 开始；连续斜杠等同一个，'.' 表示当前目录，'..' 表示返回
上级目录，不能越过根。输出没有多余斜杠或 '.'/'..' 的规范路径。
【方法】逐段扫描，以 vector 保存当前目录栈。普通名称入栈；'.' 忽略；'..'
在栈非空时弹出。最后从根目录开始拼接。例如 /a/../b//c 变成 /b/c。
时间 O(路径长度)，空间 O(路径长度)。main 内演示数个固定路径。

English: Canonicalize a Linux absolute path by collapsing slashes, ignoring '.',
and applying '..' without moving above root. Scan components and use a vector as
a directory stack, then join it with '/'. O(|path|) time and space; main contains
demonstration cases.
*/
#include <bits/stdc++.h>
using namespace std;

// 假设输入是 Linux 的绝对路径，例如 /home/user/../tmp/./a。
string simplifyPath(const string& path) {
    vector<string> directories;
    int n = static_cast<int>(path.size());
    int i = 0;

    while (i < n) {
        // 跳过连续的 '/'
        while (i < n && path[i] == '/') {
            ++i;
        }

        int start = i;
        while (i < n && path[i] != '/') {
            ++i;
        }

        if (start == i) {
            continue;
        }

        string name = path.substr(start, i - start);
        if (name == ".") {
            continue;
        }
        if (name == "..") {
            if (!directories.empty()) {
                directories.pop_back();
            }
            continue;
        }

        directories.push_back(name);
    }

    if (directories.empty()) {
        return "/";
    }

    string result;
    for (const string& directory : directories) {
        result += "/" + directory;
    }
    return result;
}

int main() {
    vector<string> paths = {
        "/home/",
        "/home//foo/",
        "/home/user/../tmp/./a/",
        "/../",
        "/a/../../b/../c//.//"
    };

    for (const string& path : paths) {
        cout << path << " -> " << simplifyPath(path) << '\n';
    }
    return 0;
}
